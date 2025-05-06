import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Added import
from torch.distributions import Categorical
from torch.utils.data import DataLoader

# Reuse components from compositional_efficiency
from compositional_efficiency.dataset import AttributeValueData
from compositional_efficiency.archs import IdentitySender, RotatedSender

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# --- REINFORCE Training Function (for Receiver) ---
def train_receiver(receiver, optimizer, messages, target_attributes, receiver_actions, rewards, entropy_coef=0.01):
    """
    Trains the Receiver using REINFORCE.
    messages: (batch_size, n_attributes) - Message symbols from sender
    target_attributes: (batch_size, n_attributes) - The original attributes
    receiver_actions: (batch_size, n_attributes) - The attributes predicted by the receiver (sampled)
    rewards: (batch_size,) - Scalar reward (1 if all attributes match, 0 otherwise)
    """
    # Forward pass to get current action probabilities
    prediction_logits = receiver(messages)

    log_probs = []
    entropies = []
    for i in range(receiver.n_attributes):
        dist = Categorical(logits=prediction_logits[i])
        log_prob = dist.log_prob(receiver_actions[:, i]) # Log prob of the action *taken*
        entropy = dist.entropy()
        log_probs.append(log_prob)
        entropies.append(entropy)

    # Sum log probs across attributes for the joint action probability
    total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1) # (batch_size,)
    # Average entropy over attributes
    total_entropy = torch.stack(entropies, dim=1).mean(dim=1) # (batch_size,)

    # Calculate policy loss (REINFORCE)
    # Normalize rewards (optional but often helpful, especially with sparse rewards)
    if rewards.std() > 1e-6: # Avoid division by zero if all rewards are the same
         rewards = (rewards - rewards.mean()) / rewards.std()
    else:
         rewards = rewards - rewards.mean() # Just center if std is zero

    policy_loss = -(total_log_prob * rewards).mean()

    # Calculate entropy loss (negative entropy, minimized)
    entropy_loss = -total_entropy.mean()

    # Total loss
    loss = policy_loss + entropy_coef * entropy_loss

    # Optimize Receiver
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(receiver.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), entropy_loss.item()

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
    return accuracy

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Train Receiver for discrete communication task using REINFORCE")
    parser.add_argument('--n_a', type=int, default=2, help="Number of attributes")
    parser.add_argument('--n_v', type=int, default=10, help="Number of values per attribute")
    parser.add_argument('--language', type=str, default='identity', choices=['identity', 'rotated'], help="Type of fixed sender")
    parser.add_argument('--receiver_emb', type=int, default=50, help='Size of the embeddings of Receiver')
    parser.add_argument('--receiver_hidden', type=int, default=100, help='Size of the hidden layer of Receiver')
    parser.add_argument('--receiver_cell', type=str, default='gru', choices=['rnn', 'gru', 'lstm'], help='RNN cell type for Receiver')
    parser.add_argument("--lr", type=float, default=0.001, help="Receiver learning rate")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient for Receiver")
    parser.add_argument("--num-steps", type=int, default=10000, help="Total training steps (batches)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluate every N steps")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--log-dir", type=str, default="./comm_logs_reinforce/", help="Directory for logs and checkpoints")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="Gradient clipping norm value")

    args = parser.parse_args()

    log_dir = args.log_dir
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data = AttributeValueData(n_attributes=args.n_a, n_values=args.n_v, mul=1, mode='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_data = AttributeValueData(n_attributes=args.n_a, n_values=args.n_v, mul=1, mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'# Train samples: {len(train_data)}, Test samples: {len(test_data)}')

    if args.language == 'identity':
        sender = IdentitySender(args.n_a, args.n_v).to(device)
    elif args.language == 'rotated':
        sender = RotatedSender(args.n_a, args.n_v).to(device)
    else:
        raise ValueError(f"Unknown language type: {args.language}")
    sender.eval()

    vocab_size = args.n_v + 1
    receiver = RnnReceiver(
        n_attributes=args.n_a,
        n_values=args.n_v,
        vocab_size=vocab_size,
        embed_dim=args.receiver_emb,
        hidden_size=args.receiver_hidden,
        cell=args.receiver_cell
    ).to(device)

    optimizer = optim.Adam(receiver.parameters(), lr=args.lr)

    print("Starting training...")
    print(f"Sender: {args.language}")
    print(f"Receiver params: {sum(p.numel() for p in receiver.parameters() if p.requires_grad)}")
    print("Using REINFORCE to train Receiver.")

    train_iterator = iter(train_loader)

    for step in range(1, args.num_steps + 1):
        receiver.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        input_attributes, _ = batch
        input_attributes = input_attributes.to(device)
        target_attributes = input_attributes

        messages, _, _ = sender(input_attributes)

        prediction_logits = receiver(messages)

        receiver_actions = []
        for i in range(receiver.n_attributes):
            dist = Categorical(logits=prediction_logits[i])
            action = dist.sample()
            receiver_actions.append(action)
        receiver_actions_tensor = torch.stack(receiver_actions, dim=1)

        matches = torch.all(receiver_actions_tensor == target_attributes, dim=1)
        rewards = matches.float().to(device)

        loss, policy_loss, entropy_loss = train_receiver(
            receiver, optimizer, messages, target_attributes, receiver_actions_tensor, rewards, args.entropy_coef
        )

        if step % 100 == 0:
            avg_reward = rewards.mean().item()
            print(f"Step {step}/{args.num_steps}, Avg Reward: {avg_reward:.4f}, Loss: {loss:.4f} (Policy: {policy_loss:.4f}, Entropy: {entropy_loss:.4f})")

        if step % args.eval_interval == 0:
            receiver.eval()
            accuracy = evaluate_communication(sender, receiver, test_loader)
            print(f"--- Step {step} Evaluation Accuracy: {accuracy:.4f} ---")

        if step % args.checkpoint_interval == 0 or step == args.num_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"receiver_step_{step}.pt")
            torch.save({
                'step': step,
                'receiver_state_dict': receiver.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")

if __name__ == "__main__":
    main()