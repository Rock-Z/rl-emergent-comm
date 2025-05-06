import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

# Reuse components from compositional_efficiency
from compositional_efficiency.dataset import AttributeValueData
from compositional_efficiency.archs import IdentitySender, RotatedSender

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Learned Sender Model ---
class RnnSender(nn.Module):
    def __init__(self, n_attributes, n_values, vocab_size, embed_dim, hidden_size, cell='gru', max_len=2):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.input_dim = n_attributes * n_values # Input is flattened one-hot
        self.vocab_size = vocab_size # Output vocab size (n_values + 1 for EOS/padding)
        self.embed_dim = embed_dim # Embedding for output symbols (used as input to RNN step)
        self.hidden_size = hidden_size
        self.max_len = max_len # Fixed message length

        # Layer to get initial hidden state from flattened one-hot input
        self.input_to_hidden = nn.Linear(self.input_dim, hidden_size)

        # Embedding for the output symbols (used as input during generation)
        # vocab_size needs to account for symbols 1..n_values and potentially 0 (SOS/PAD)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN cell
        rnn_cell_type = nn.GRUCell if cell.lower() == 'gru' else \
                        nn.LSTMCell if cell.lower() == 'lstm' else \
                        nn.RNNCell
        self.rnn_cell = rnn_cell_type(embed_dim, hidden_size)
        self.is_lstm = isinstance(self.rnn_cell, nn.LSTMCell)

        # Output layer from hidden state to vocab logits
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        # Learned start-of-sequence symbol embedding (using index 0)
        # We don't use a separate parameter but use index 0 of the embedding table
        # self.sos_embedding = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, x):
        """
        Input x: (batch_size, n_attributes) tensor of attribute values (0 to n_values-1)
        Output message: (batch_size, max_len) tensor of symbols (1 to n_values)
        Output log_probs: (batch_size, max_len) tensor of log probabilities for the chosen symbols
        Output entropy: (batch_size, max_len) tensor of entropy for the distributions at each step
        """
        batch_size = x.size(0)
        device = x.device

        # 1. Convert input attributes to one-hot and get initial hidden state
        x_one_hot = F.one_hot(x, num_classes=self.n_values).view(batch_size, -1).float()
        h = self.input_to_hidden(x_one_hot) # (batch_size, hidden_size)
        c = torch.zeros_like(h) if self.is_lstm else None # Initial cell state for LSTM

        # 2. Generate sequence step-by-step
        messages = []
        log_probs = []
        entropies = []

        # Initial input embedding: Use embedding of index 0 as SOS symbol
        step_input_emb = self.embedding(torch.zeros(batch_size, dtype=torch.long, device=device))

        for _ in range(self.max_len):
            # Run RNN cell
            if self.is_lstm:
                h, c = self.rnn_cell(step_input_emb, (h, c))
            else:
                h = self.rnn_cell(step_input_emb, h)

            # Get logits over vocabulary
            step_logits = self.hidden_to_output(h) # (batch_size, vocab_size)

            # Mask the logit for index 0 (SOS/PAD symbol) to prevent sampling it
            # Set its logit to a very large negative number
            masked_logits = step_logits.clone()
            masked_logits[:, 0] = -float('inf')

            # Sample symbol (use Categorical distribution with masked logits)
            dist = Categorical(logits=masked_logits)
            symbol = dist.sample() # (batch_size,)

            # Store results using the original (unmasked) distribution for correct log_prob and entropy
            # Calculate log_prob and entropy based on the *original* logits
            original_dist = Categorical(logits=step_logits)
            log_prob_for_symbol = original_dist.log_prob(symbol)
            entropy_for_step = original_dist.entropy()

            messages.append(symbol)
            log_probs.append(log_prob_for_symbol)
            entropies.append(entropy_for_step)

            # Prepare input embedding for next step (use embedding of the generated symbol)
            step_input_emb = self.embedding(symbol) # symbol is already in [1, vocab_size-1]

        # Stack results into tensors
        message_tensor = torch.stack(messages, dim=1) # (batch_size, max_len)
        log_prob_tensor = torch.stack(log_probs, dim=1) # (batch_size, max_len)
        entropy_tensor = torch.stack(entropies, dim=1) # (batch_size, max_len)

        # The message tensor now naturally contains symbols >= 1 because 0 was masked.
        # No need for clamping anymore.
        # message_output_tensor = message_tensor.clamp(min=1)

        # Return the generated message, log_probs and entropies.
        return message_tensor, log_prob_tensor, entropy_tensor


# --- Receiver Model (Agent Wrapper) ---
class RnnReceiver(nn.Module):
    """
    Wrapper around the core Receiver from archs.py to handle message sequences.
    Takes a message sequence, processes it with RNN, and uses CoreReceiver
    to predict attributes from the final hidden state.
    """
    def __init__(self, n_attributes, n_values, vocab_size, embed_dim, hidden_size, cell='gru'):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        rnn_cell_type = nn.GRU if cell.lower() == 'gru' else \
                        nn.LSTM if cell.lower() == 'lstm' else \
                        nn.RNN
        self.rnn = rnn_cell_type(embed_dim, hidden_size, batch_first=True)

        self.output_module = nn.Linear(hidden_size, n_attributes * n_values)

    def forward(self, message):
        """
        Input message: Batch of message symbol indices, shape (batch_size, message_len)
                       In this case, message_len == n_attributes
        Output prediction_logits: List of tensors, one per attribute head.
                                  Each tensor shape: (batch_size, n_values)
        """
        batch_size = message.shape[0]

        # Ensure message symbols are within embedding range [0, vocab_size-1]
        # Sender outputs symbols in range [1, n_values], so vocab_size must be n_values + 1
        if torch.any(message >= self.vocab_size) or torch.any(message < 0):
             raise ValueError(f"Message symbols out of range [0, {self.vocab_size-1}], got min {message.min()} max {message.max()}")

        embedded = self.embedding(message) # (batch_size, n_attributes, embed_dim)

        # Process sequence with RNN
        if isinstance(self.rnn, nn.LSTM):
            rnn_out, (h_n, c_n) = self.rnn(embedded)
            last_hidden = h_n[-1] # Use hidden state from last layer
        else: # GRU or RNN
            rnn_out, h_n = self.rnn(embedded)
            last_hidden = h_n[-1] # Use hidden state from last layer
        # rnn_out: (batch_size, n_attributes, hidden_size)
        # last_hidden: (batch_size, hidden_size)

        # Pass final hidden state to the output linear layer
        output_flat = self.output_module(last_hidden) # (batch_size, n_a * n_v)

        # Reshape output to have separate logits for each attribute
        output_logits_per_attribute = output_flat.view(batch_size, self.n_attributes, self.n_values)

        # Return list of logits per attribute for Categorical distribution
        # Transpose to (n_attributes, batch_size, n_values) then split
        # Squeeze is needed if batch_size is 1 during eval? No, list comprehension handles it.
        return [output_logits_per_attribute[:, i, :] for i in range(self.n_attributes)]


# --- Unified Training Step Function ---
def train_step(sender, receiver, sender_optimizer, receiver_optimizer,
               input_attributes,
               receiver_actions_tensor, rewards,
               sender_entropy_coef, receiver_entropy_coef, grad_norm,
               train_sender_flag):
    """
    Performs one training step for Receiver and optionally Sender using REINFORCE.
    Generates the message within this step.
    Returns losses and the generated message details.
    """
    receiver_loss_val, receiver_pol_loss_val, receiver_ent_loss_val = 0.0, 0.0, 0.0
    sender_loss_val, sender_pol_loss_val, sender_ent_loss_val = 0.0, 0.0, 0.0

    # --- Interaction: Sender produces message ---
    # Set sender mode based on whether it's being trained
    if not train_sender_flag:
        sender.eval()
        with torch.no_grad(): # Ensure no grads for fixed sender forward pass
             messages, sender_log_probs, sender_entropies = sender(input_attributes)
    else:
        sender.train() # Ensure sender is in train mode for gradient calculation
        messages, sender_log_probs, sender_entropies = sender(input_attributes)

    # --- Train Receiver ---
    receiver.train()
    # Detach messages before feeding to receiver to prevent grads flowing back
    # through message generation during receiver optimization step.
    prediction_logits = receiver(messages.detach())

    log_probs = []
    entropies = []
    for i in range(receiver.n_attributes):
        dist = Categorical(logits=prediction_logits[i])
        log_prob = dist.log_prob(receiver_actions_tensor[:, i]) # Log prob of the action *taken*
        entropy = dist.entropy()
        log_probs.append(log_prob)
        entropies.append(entropy)

    # Sum log probs across attributes for the joint action probability
    receiver_total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1) # (batch_size,)
    # Average entropy over attributes
    receiver_total_entropy = torch.stack(entropies, dim=1).mean(dim=1) # (batch_size,)

    # Normalize rewards (optional but often helpful, especially with sparse rewards)
    # if rewards.std() > 1e-6: # Avoid division by zero if all rewards are the same
    #      rewards = (rewards - rewards.mean()) / rewards.std()
    # else:
    #      rewards = rewards - rewards.mean() # Just center if std is zero

    receiver_policy_loss = -(receiver_total_log_prob * rewards).mean()
    receiver_entropy_loss = -receiver_total_entropy.mean()
    receiver_loss = receiver_policy_loss + receiver_entropy_coef * receiver_entropy_loss

    receiver_optimizer.zero_grad()
    receiver_loss.backward()
    torch.nn.utils.clip_grad_norm_(receiver.parameters(), max_norm=grad_norm)
    receiver_optimizer.step()

    receiver_loss_val = receiver_loss.item()
    receiver_pol_loss_val = receiver_policy_loss.item()
    receiver_ent_loss_val = receiver_entropy_loss.item()

    # --- Train Sender (if learned) ---
    if train_sender_flag:
        # sender is already in train() mode from message generation above
        # Calculate sender loss components using REINFORCE
        # sender_log_probs and sender_entropies were calculated during the forward pass
        sender_total_log_prob = sender_log_probs.sum(dim=1) # (batch_size,)
        sender_total_entropy = sender_entropies.mean(dim=1) # (batch_size,)

        # Use the same normalized rewards as the receiver
        # norm_rewards calculated above

        sender_policy_loss = -(sender_total_log_prob * rewards).mean()
        sender_entropy_loss = -sender_total_entropy.mean()
        sender_loss = sender_policy_loss + sender_entropy_coef * sender_entropy_loss

        # Optimize Sender
        sender_optimizer.zero_grad()
        sender_loss.backward()
        torch.nn.utils.clip_grad_norm_(sender.parameters(), max_norm=grad_norm)
        sender_optimizer.step()

        sender_loss_val = sender_loss.item()
        sender_pol_loss_val = sender_policy_loss.item()
        sender_ent_loss_val = sender_entropy_loss.item()

    # Return losses AND message details (needed for receiver action sampling in main loop)
    return (receiver_loss_val, receiver_pol_loss_val, receiver_ent_loss_val,
            sender_loss_val, sender_pol_loss_val, sender_ent_loss_val,
            messages, sender_log_probs, sender_entropies)


# --- Evaluation Function ---
def evaluate_communication(sender, receiver, data_loader):
    """Evaluates the communication accuracy using a DataLoader."""
    sender.eval()
    receiver.eval()
    correct_predictions = 0
    total_samples = 0
    printed_examples = 0
    max_examples_to_print = 5 # Limit how many examples we print per evaluation

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_attributes, _ = batch
            input_attributes = input_attributes.to(device)
            batch_size = input_attributes.shape[0]

            messages, _, _ = sender(input_attributes)
            prediction_logits = receiver(messages)

            # Get receiver predictions (argmax for evaluation)
            predictions = []
            for i in range(receiver.n_attributes):
                pred = torch.argmax(prediction_logits[i], dim=1)
                predictions.append(pred)
            receiver_output = torch.stack(predictions, dim=1)

            # Compare with original input attributes
            matches = torch.all(receiver_output == input_attributes, dim=1)
            correct_predictions += matches.sum().item()
            total_samples += batch_size

            # Print some examples from the first batch of this evaluation
            if batch_idx == 0 and printed_examples < max_examples_to_print:
                num_to_print = min(max_examples_to_print - printed_examples, batch_size)
                print("--- Evaluation Examples ---")
                for i in range(num_to_print):
                    print(f"  Input:    {input_attributes[i].cpu().numpy()}")
                    print(f"  Message:  {messages[i].cpu().numpy()}")
                    print(f"  Predicted:{receiver_output[i].cpu().numpy()}")
                    print(f"  Match:    {matches[i].item()}")
                    print("  -----")
                    printed_examples += 1
                if printed_examples >= max_examples_to_print:
                    print("-------------------------")

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    sender.train()
    receiver.train() 
    return accuracy

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Train Receiver for discrete communication task using REINFORCE")
    parser.add_argument('--n_a', type=int, default=2, help="Number of attributes")
    parser.add_argument('--n_v', type=int, default=10, help="Number of values per attribute")
    parser.add_argument('--sender_type', type=str, default='identity', choices=['identity', 'rotated', 'learned'], help="Type of sender")
    parser.add_argument('--max_len', type=int, default=None, help="Message length (defaults to n_a)")

    # Sender params (only used if sender_type=='learned')
    parser.add_argument('--sender_emb', type=int, default=50, help='Size of the embeddings of Sender')
    parser.add_argument('--sender_hidden', type=int, default=100, help='Size of the hidden layer of Sender')
    parser.add_argument('--sender_cell', type=str, default='gru', choices=['rnn', 'gru', 'lstm'], help='RNN cell type for Sender')
    parser.add_argument("--sender_lr", type=float, default=0.001, help="Sender learning rate")
    parser.add_argument("--sender_entropy_coef", type=float, default=0.1, help="Entropy coefficient for Sender")

    # Receiver params
    parser.add_argument('--receiver_emb', type=int, default=50, help='Size of the embeddings of Receiver')
    parser.add_argument('--receiver_hidden', type=int, default=100, help='Size of the hidden layer of Receiver')
    parser.add_argument('--receiver_cell', type=str, default='gru', choices=['rnn', 'gru', 'lstm'], help='RNN cell type for Receiver')
    parser.add_argument("--receiver_lr", type=float, default=0.001, help="Receiver learning rate")
    parser.add_argument("--receiver_entropy_coef", type=float, default=0.1, help="Entropy coefficient for Receiver")

    # General training params
    parser.add_argument("--num-steps", type=int, default=10000, help="Total training steps (batches)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--log-dir", type=str, default="./comm_logs_reinforce/", help="Directory for logs and checkpoints")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="Gradient clipping norm value")

    args = parser.parse_args()

    # Set max_len default if not provided
    if args.max_len is None:
        args.max_len = args.n_a
        print(f"Message max_len not specified, defaulting to n_a: {args.max_len}")

    log_dir = args.log_dir
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data = AttributeValueData(n_attributes=args.n_a, n_values=args.n_v, mul=1, mode='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_data = AttributeValueData(n_attributes=args.n_a, n_values=args.n_v, mul=1, mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'# Train samples: {len(train_data)}, Test samples: {len(test_data)}')

    # Vocab size needs to accommodate symbols 1..n_v and potentially 0 for SOS/PAD
    vocab_size = args.n_v + 1

    # --- Initialize Sender ---
    sender_optimizer = None
    train_sender_flag = (args.sender_type == 'learned') # Flag to control sender training
    if args.sender_type == 'identity':
        sender = IdentitySender(args.n_a, args.n_v).to(device)
        sender.eval() # Fixed senders don't train
    elif args.sender_type == 'rotated':
        sender = RotatedSender(args.n_a, args.n_v).to(device)
        sender.eval() # Fixed senders don't train
    elif args.sender_type == 'learned':
        sender = RnnSender(
            n_attributes=args.n_a,
            n_values=args.n_v,
            vocab_size=vocab_size,
            embed_dim=args.sender_emb,
            hidden_size=args.sender_hidden,
            cell=args.sender_cell,
            max_len=args.max_len
        ).to(device)
        sender_optimizer = optim.Adam(sender.parameters(), lr=args.sender_lr)
        print(f"Sender params (learned): {sum(p.numel() for p in sender.parameters() if p.requires_grad)}")
    else:
        raise ValueError(f"Unknown sender type: {args.sender_type}")

    # --- Initialize Receiver ---
    receiver = RnnReceiver(
        n_attributes=args.n_a,
        n_values=args.n_v,
        vocab_size=vocab_size, # Receiver needs same vocab size
        embed_dim=args.receiver_emb,
        hidden_size=args.receiver_hidden,
        cell=args.receiver_cell
    ).to(device)
    receiver_optimizer = optim.Adam(receiver.parameters(), lr=args.receiver_lr)

    print("Starting training...")
    print(f"Sender type: {args.sender_type}")
    print(f"Receiver params: {sum(p.numel() for p in receiver.parameters() if p.requires_grad)}")
    print(f"Using REINFORCE to train {'Receiver and Sender' if train_sender_flag else 'Receiver'}.")

    train_iterator = iter(train_loader)
    sender_loss_hist, sender_pol_loss_hist, sender_ent_loss_hist = [], [], []
    receiver_loss_hist, receiver_pol_loss_hist, receiver_ent_loss_hist = [], [], []
    reward_hist = []

    for step in range(1, args.num_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        input_attributes, _ = batch
        input_attributes = input_attributes.to(device)
        target_attributes = input_attributes # Receiver's target is the original input

        # --- Interaction: Sample Receiver Actions ---
        # Generate message to feed to receiver for action sampling.
        # Ensure correct sender mode for this temporary generation.
        if not train_sender_flag:
            sender.eval()
            with torch.no_grad():
                 messages_for_sampling, _, _ = sender(input_attributes)
        else:
            sender.train() # Need train mode if sender is learned, even for sampling pass
            # with torch.no_grad(): # Still no grads needed for sampling itself
            messages_for_sampling, _, _ = sender(input_attributes)

        # Receiver processes message and samples actions
        receiver.eval() # Use eval mode for sampling actions
        with torch.no_grad(): # No grads needed for sampling actions
            prediction_logits = receiver(messages_for_sampling)
            # Sample receiver actions for REINFORCE
            receiver_actions = []
            for i in range(receiver.n_attributes):
                dist = Categorical(logits=prediction_logits[i])
                action = dist.sample() # (batch_size,)
                receiver_actions.append(action)
            receiver_actions_tensor = torch.stack(receiver_actions, dim=1) # (batch_size, n_attributes)

        # --- Calculate Reward ---
        matches = torch.all(receiver_actions_tensor == target_attributes, dim=1)
        rewards = matches.float().to(device) # (batch_size,)

        # --- Training Step (includes message generation) ---
        rec_loss, rec_pol_loss, rec_ent_loss, \
        send_loss, send_pol_loss, send_ent_loss, \
        messages, sender_log_probs, sender_entropies = train_step(
            sender=sender,
            receiver=receiver,
            sender_optimizer=sender_optimizer,
            receiver_optimizer=receiver_optimizer,
            input_attributes=input_attributes,
            receiver_actions_tensor=receiver_actions_tensor,
            rewards=rewards.detach(),
            sender_entropy_coef=args.sender_entropy_coef,
            receiver_entropy_coef=args.receiver_entropy_coef,
            grad_norm=args.grad_norm,
            train_sender_flag=train_sender_flag
        )

        # Store losses
        receiver_loss_hist.append(rec_loss)
        receiver_pol_loss_hist.append(rec_pol_loss)
        receiver_ent_loss_hist.append(rec_ent_loss)
        sender_loss_hist.append(send_loss)
        sender_pol_loss_hist.append(send_pol_loss)
        sender_ent_loss_hist.append(send_ent_loss)
        reward_hist.append(rewards.mean().item())

        # --- Logging ---
        if step % 100 == 0:
            avg_reward = torch.tensor(reward_hist[-100:]).mean().item()
            avg_rec_loss = torch.tensor(receiver_loss_hist[-100:]).mean().item()
            avg_rec_pol = torch.tensor(receiver_pol_loss_hist[-100:]).mean().item()
            avg_rec_ent = torch.tensor(receiver_ent_loss_hist[-100:]).mean().item()

            log_msg = (f"Step {step}/{args.num_steps}, Avg Reward: {avg_reward:.4f}, "
                       f"Rec Loss: {avg_rec_loss:.4f} (Pol: {avg_rec_pol:.4f}, Ent: {avg_rec_ent:.4f})")

        sender_ent_loss_hist.append(send_ent_loss)
        reward_hist.append(rewards.mean().item()) # Store raw reward mean

        # --- Logging ---
        if step % 100 == 0:
            avg_reward = torch.tensor(reward_hist[-100:]).mean().item()
            avg_rec_loss = torch.tensor(receiver_loss_hist[-100:]).mean().item()
            avg_rec_pol = torch.tensor(receiver_pol_loss_hist[-100:]).mean().item()
            avg_rec_ent = torch.tensor(receiver_ent_loss_hist[-100:]).mean().item()

            log_msg = (f"Step {step}/{args.num_steps}, Avg Reward: {avg_reward:.4f}, "
                       f"Rec Loss: {avg_rec_loss:.4f} (Pol: {avg_rec_pol:.4f}, Ent: {avg_rec_ent:.4f})")

            if train_sender_flag:
                avg_send_loss = torch.tensor(sender_loss_hist[-100:]).mean().item()
                avg_send_pol = torch.tensor(sender_pol_loss_hist[-100:]).mean().item()
                avg_send_ent = torch.tensor(sender_ent_loss_hist[-100:]).mean().item()
                log_msg += (f", Send Loss: {avg_send_loss:.4f} (Pol: {avg_send_pol:.4f}, Ent: {avg_send_ent:.4f})")

            print(log_msg)

        # --- Evaluation ---
        if step % args.eval_interval == 0:
            # Ensure models are in eval mode for evaluation
            sender.eval()
            receiver.eval()
            accuracy = evaluate_communication(sender, receiver, test_loader)
            print(f"--- Step {step} Evaluation Accuracy: {accuracy:.4f} ---")
            # Restore train mode after evaluation (will be set correctly at start of next loop iter)


        # --- Checkpointing ---
        if step % args.checkpoint_interval == 0 or step == args.num_steps:
            save_obj = {
                'step': step,
                'receiver_state_dict': receiver.state_dict(),
                'receiver_optimizer_state_dict': receiver_optimizer.state_dict(),
                'args': args
            }
            if train_sender_flag and sender_optimizer is not None: # Check optimizer exists
                save_obj['sender_state_dict'] = sender.state_dict()
                save_obj['sender_optimizer_state_dict'] = sender_optimizer.state_dict()

            checkpoint_path = os.path.join(checkpoint_dir, f"agent_step_{step}.pt") # More general name
            torch.save(save_obj, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

if __name__ == "__main__":
    main()