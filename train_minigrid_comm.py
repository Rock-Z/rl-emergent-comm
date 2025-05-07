import argparse
# import os # Unused
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import sys # Import sys for sys.argv

import egg.core as core
from egg.core import EarlyStopperAccuracy

# Imports from existing project files
from train_minigrid import RNNAgent as NavRNNAgentBase # Will be wrapped or modified
from train_minigrid import make_env # Reusing make_env
# custom_minigrid_env is imported by train_minigrid, which should register the env.

# Ensure device is set up early
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Sender Agent
class CoordinateSender(nn.Module):
    """ Agent that takes (x,y) coordinates and outputs a hidden state for RNN message generator. """
    def __init__(self, n_coord_features, sender_rnn_hidden_dim):
        super().__init__()
        self.fc = nn.Linear(n_coord_features, sender_rnn_hidden_dim)

    def forward(self, x):
        # x is (batch_size, n_coord_features), e.g., (batch_size, 2) for (x,y)
        return self.fc(x)

# 2. Navigation Agent (modified from train_minigrid.RNNAgent)
class NavRNNAgent(NavRNNAgentBase):
    """ Modified RNNAgent to accept and return hidden state. """
    def __init__(self, obs_space, num_actions, hidden_size=64, direction_embed_size=8):
        super().__init__(obs_space, num_actions, hidden_size, direction_embed_size)
        # self.hidden is no longer used internally in fwd pass for single step
        # but reset_hidden can be used to get a zero hidden state if needed.
        # The actual hidden state will be passed explicitly in forward.
        self.nav_hidden_size = hidden_size # Store for clarity

    def reset_hidden(self, batch_size=1):
        # Provides a default zero hidden state if needed
        return torch.zeros(batch_size, self.nav_hidden_size, device=device)

    def forward(self, obs_dict, hidden_state):
        # obs_dict contains 'image' and 'direction' tensors (already batched)
        # hidden_state is (batch_size, nav_hidden_size)
        img = obs_dict['image'] 
        direction = obs_dict['direction']

        # Assuming single step, not sequence.
        # img: (batch_size, H, W, C), direction: (batch_size,)
        img = img / 255.0
        img = img.permute(0, 3, 1, 2) # (batch_size, C, H, W)
        conv_out = self.conv(img)

        direction_embedded = self.direction_embedding(direction)
        combined_features = torch.cat((conv_out, direction_embedded), dim=1)
        
        next_hidden_state = self.rnn(combined_features, hidden_state)
        
        policy_logits = self.actor(next_hidden_state)
        values = self.critic(next_hidden_state) # values will be (batch_size, 1)
        
        return policy_logits, values.squeeze(-1), next_hidden_state


# 3. EGG Receiver Agent (Message Decoder)
class MessageDecoder(nn.Module):
    """ Decodes message embedding from EGG's RnnEncoder into initial_nav_hidden_state. """
    def __init__(self, message_embedding_dim, nav_rnn_hidden_dim):
        super().__init__()
        self.fc = nn.Linear(message_embedding_dim, nav_rnn_hidden_dim)

    def forward(self, message_embedding, _receiver_input=None):
        # message_embedding is (batch_size, message_embedding_dim)
        # _receiver_input is not used by this decoder directly.
        return self.fc(message_embedding)


# 4. Custom EGG Loss Function
class MiniGridNavigationLoss(nn.Module):
    def __init__(self, nav_agent_instance, nav_agent_optimizer, env_factory, 
                 max_episode_steps, gamma, nav_entropy_coeff):
        super().__init__()
        self.nav_agent = nav_agent_instance
        self.nav_agent_optimizer = nav_agent_optimizer
        self.env_factory = env_factory # Function to create a new env instance
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.nav_entropy_coeff = nav_entropy_coeff

    def forward(self, sender_input_coords, _message, receiver_initial_obs_dict, 
                initial_nav_hidden_state, _labels):
        # sender_input_coords: (batch_size, 2) - used for potential env setup or logging
        # _message: raw message from sender (not directly used here, already processed by EGG Receiver)
        # receiver_initial_obs_dict: {'image': (B,H,W,C), 'direction': (B,)}
        # initial_nav_hidden_state: (batch_size, nav_hidden_size) - output of EGG Receiver (MessageDecoder)
        # _labels: not used

        batch_size = sender_input_coords.size(0)
        cumulative_reward_for_egg_loss = torch.zeros(batch_size, device=sender_input_coords.device)
        
        batch_episode_rewards = []
        batch_nav_agent_losses = []

        for i in range(batch_size):
            # For each item in batch, run an episode
            env_instance = self.env_factory() # Create a fresh env for each item
            
            # Reset this specific env_instance to get its actual starting observation
            actual_initial_obs_raw, _ = env_instance.reset()

            # Prepare the first observation for the nav_agent from this env_instance's reset
            current_obs_img = torch.FloatTensor(actual_initial_obs_raw['image']).unsqueeze(0).to(device) # (1,H,W,C)
            current_obs_dir = torch.tensor([actual_initial_obs_raw['direction']], dtype=torch.long, device=device) # (1,)
            current_nav_obs_dict = {'image': current_obs_img, 'direction': current_obs_dir}
            
            current_hidden = initial_nav_hidden_state[i].unsqueeze(0) # (1, nav_hidden_size)

            ep_rewards, ep_log_probs, ep_values, ep_entropies, ep_masks = [], [], [], [], []

            for _step in range(self.max_episode_steps):
                policy_logits, value, next_hidden = self.nav_agent(current_nav_obs_dict, current_hidden)
                
                dist = Categorical(logits=policy_logits)
                action = dist.sample() # action is a tensor e.g. tensor([int_action])
                
                log_prob_action = dist.log_prob(action)
                entropy_action = dist.entropy()

                # Perform step in environment
                # action.item() is an int
                next_obs_raw_dict, reward, terminated, truncated, _info = env_instance.step(action.item())
                done = terminated or truncated

                ep_rewards.append(reward)
                ep_log_probs.append(log_prob_action)
                ep_values.append(value) # value from nav_agent is already 1D for this item
                ep_entropies.append(entropy_action)
                ep_masks.append(1.0 - float(done))

                # Prepare next state
                current_nav_obs_dict = {
                    'image': torch.FloatTensor(next_obs_raw_dict['image']).unsqueeze(0).to(device),
                    'direction': torch.tensor([next_obs_raw_dict['direction']], dtype=torch.long).to(device)
                }
                current_hidden = next_hidden
                
                if done:
                    break
            
            env_instance.close()

            # Train NavRNNAgent for this episode
            returns = []
            R = 0.0
            for r_idx in reversed(range(len(ep_rewards))):
                R = ep_rewards[r_idx] + self.gamma * R * ep_masks[r_idx]
                returns.insert(0, R)
            
            returns_t = torch.tensor(returns, dtype=torch.float, device=device)
            log_probs_t = torch.cat(ep_log_probs)
            values_t = torch.cat(ep_values) # values were (1,), cat makes (T,)
            entropies_t = torch.cat(ep_entropies)

            advantage = returns_t - values_t.detach() # Detach values for actor update
            
            actor_loss = -(log_probs_t * advantage).mean()
            critic_loss = F.mse_loss(values_t, returns_t) # Don't detach returns_t
            entropy_bonus = -self.nav_entropy_coeff * entropies_t.mean()
            
            nav_agent_loss = actor_loss + 0.5 * critic_loss + entropy_bonus
            
            # Update NavRNNAgent only if the loss module itself is in training mode
            # AND gradients are globally enabled (i.e., not in a torch.no_grad() context)
            if self.training and torch.is_grad_enabled():
                self.nav_agent_optimizer.zero_grad()
                nav_agent_loss.backward(retain_graph=True) # Grads for NavAgent params, retain graph for EGG
                self.nav_agent_optimizer.step()
            
            # Loss for EGG game (communication part) = negative sum of rewards for this episode
            cumulative_reward_for_egg_loss[i] = -torch.tensor(sum(ep_rewards), dtype=torch.float, device=device)
            
            batch_episode_rewards.append(sum(ep_rewards))
            batch_nav_agent_losses.append(nav_agent_loss.item())

        # Average loss for EGG framework (across batch)
        # This loss tensor is what SenderReceiverRnnReinforce uses.
        # Gradients will flow from this loss back to initial_nav_hidden_state,
        # then to MessageDecoder, its RnnEncoder, and then to Sender.
        final_loss_for_egg_framework = cumulative_reward_for_egg_loss 
        # EGG expects loss per batch item, not mean. (batch_size,)

        aux_info = {
            'acc': torch.tensor(np.mean(batch_episode_rewards), device=device), # 'acc' is convention for EGG
            'reward_mean': np.mean(batch_episode_rewards),
            'nav_loss_mean': np.mean(batch_nav_agent_losses),
            'reward_std': np.std(batch_episode_rewards),
        }
        return final_loss_for_egg_framework, aux_info


# 5. Data Handling
class MiniGridEpisodeDataset(Dataset):
    def __init__(self, env_factory_fn, num_episodes_per_epoch, grid_size=10):
        self.env_factory = env_factory_fn
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.grid_size = grid_size 

    def __len__(self):
        return self.num_episodes_per_epoch

    def __getitem__(self, idx):
        env = self.env_factory()
        # Reset returns (obs_dict, info_dict) for gymnasium
        obs_dict, info = env.reset() 
        
        # Agent position is available after reset
        agent_pos_x, agent_pos_y = env.agent_pos 
        
        # Sender input: (x,y) coordinates normalized or as features
        # Using raw coords as a 2-element tensor.
        # Normalize by grid_size for potentially better learning
        sender_input = torch.tensor(
            [float(agent_pos_x) / self.grid_size, float(agent_pos_y) / self.grid_size], 
            dtype=torch.float
        )
        
        # Receiver input: initial observation dictionary (convert numpy to tensors)
        receiver_input_img = torch.FloatTensor(obs_dict['image']) # (H, W, C)
        receiver_input_dir = torch.tensor(obs_dict['direction'], dtype=torch.long)
        
        env.close()
        
        labels = torch.zeros(1) # Dummy labels
        
        return sender_input, labels, {'image': receiver_input_img, 'direction': receiver_input_dir}

def collate_fn_minigrid(batch_list):
    sender_inputs, labels, receiver_input_dicts = zip(*batch_list)
    
    sender_inputs_b = torch.stack(sender_inputs)
    labels_b = torch.stack(labels) # Corrected: labels_b -> labels
    
    images_b = torch.stack([d['image'] for d in receiver_input_dicts])
    directions_b = torch.stack([d['direction'] for d in receiver_input_dicts])
    receiver_inputs_b = {'image': images_b, 'direction': directions_b}
    
    return sender_inputs_b, labels_b, receiver_inputs_b

# 6. Main Training Function
def get_params(params):
    parser = argparse.ArgumentParser()
    # Arguments also defined in compo_vs_generalization/train.py (or similar pattern)
    parser.add_argument('--sender_cell', type=str, default='gru', help='Sender RNN cell type (rnn, gru, lstm)')
    parser.add_argument('--receiver_cell', type=str, default='gru', help='Receiver RNN cell type')
    parser.add_argument('--sender_hidden', type=int, default=64, help='Sender RNN hidden size')
    parser.add_argument('--receiver_hidden', type=int, default=64, help='Receiver RNN hidden size (for message encoder)')
    parser.add_argument('--sender_emb', type=int, default=32, help='Sender embedding dimension') # Changed from sender_embedding_dim
    parser.add_argument('--receiver_emb', type=int, default=32, help='Receiver embedding dimension') # Changed from receiver_embedding_dim
    parser.add_argument('--sender_entropy_coeff', type=float, default=0.01, help='Sender entropy coefficient')
    
    # Arguments specific to this script or EGG arguments that compo_vs_generalization/train.py also defines/customizes
    parser.add_argument('--early_stopping_threshold', type=float, default=0.95, help="Early stopping threshold for reward_mean")
    parser.add_argument('--episodes_per_epoch', type=int, default=1000, help="Number of episodes in one epoch")
    parser.add_argument('--length_cost', type=float, default=0.0, help="Cost for message length")

    # Navigation agent specific arguments (custom to this script)
    parser.add_argument('--nav_agent_hidden_size', type=int, default=64, help='Navigation agent RNN hidden size')
    parser.add_argument('--nav_agent_lr', type=float, default=5e-4, help='Learning rate for Navigation agent optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for navigation task')
    parser.add_argument('--nav_entropy_coeff', type=float, default=0.01, help='Navigation agent entropy coefficient')
    parser.add_argument('--max_episode_steps', type=int, default=100, help='Max steps per MiniGrid episode')
    
    # Arguments like --lr, --batch_size, --n_epochs, --vocab_size, --max_len
    # will be added by core.init if not present.
    # We remove them from here to match compo_vs_generalization/train.py's style
    # and to avoid conflicts with core.init's _populate_cl_params.

    args = core.init(arg_parser=parser, params=params)
    return args

def main(params):
    opts = get_params(params) # Use the new get_params function
    print(f"Using device: {device}")
    print(f"Running with options: {opts}")

    # Environment Factory
    # make_env is imported from train_minigrid.py
    # It already registers "MiniGrid-Empty-Random-10x10-v0" if custom_minigrid_env.py is imported.
    # It also applies wrappers like RGBImgPartialObsWrapper and RecordEpisodeStatistics
    def env_factory():
        # Ensure no video recording during dataset generation or normal training runs
        # Video recording can be done separately for evaluation if needed.
        return make_env(video_folder=None) 

    # Create a sample env to get observation and action spaces
    sample_env = env_factory()
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    num_actions = action_space.n
    grid_size = sample_env.width # Assuming square grid, e.g. 10 for 10x10
    sample_env.close()

    # Sender
    coord_sender_agent = CoordinateSender(n_coord_features=2, sender_rnn_hidden_dim=opts.sender_hidden).to(device)
    sender = core.RnnSenderReinforce(
        agent=coord_sender_agent,
        vocab_size=opts.vocab_size, # Provided by core.init
        embed_dim=opts.sender_emb, # Changed from sender_embedding_dim to sender_emb
        hidden_size=opts.sender_hidden, 
        max_len=opts.max_len, # Provided by core.init
        force_eos=True, 
        cell=opts.sender_cell.upper() 
    ).to(device)

    # Navigation Agent (and its optimizer)
    nav_agent = NavRNNAgent(
        obs_space=obs_space, 
        num_actions=num_actions, 
        hidden_size=opts.nav_agent_hidden_size # Ensure this param exists
    ).to(device)
    nav_agent_optimizer = optim.Adam(nav_agent.parameters(), lr=opts.nav_agent_lr) # Ensure lr param

    # EGG Receiver (Message Decoder)
    # The RnnReceiverDeterministic's RnnEncoder will have hidden_size = opts.receiver_hidden
    # This becomes the input dim for MessageDecoder.
    message_decoder_agent = MessageDecoder(
        message_embedding_dim=opts.receiver_hidden, 
        nav_rnn_hidden_dim=opts.nav_agent_hidden_size 
    ).to(device)
    receiver = core.RnnReceiverDeterministic(
        agent=message_decoder_agent,
        vocab_size=opts.vocab_size, # Provided by core.init
        embed_dim=opts.receiver_emb, # Changed from receiver_embedding_dim to receiver_emb
        hidden_size=opts.receiver_hidden, 
        cell=opts.receiver_cell.upper() 
    ).to(device)

    # Loss Function
    minigrid_loss = MiniGridNavigationLoss(
        nav_agent_instance=nav_agent,
        nav_agent_optimizer=nav_agent_optimizer,
        env_factory=env_factory,
        max_episode_steps=opts.max_episode_steps, # Ensure this param
        gamma=opts.gamma, # Discount factor for NavAgent
        nav_entropy_coeff=opts.nav_entropy_coeff # Ensure this param
    ).to(device)

    # EGG Game
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        minigrid_loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0, # RnnReceiverDeterministic has no entropy term for its policy
        length_cost=opts.length_cost if hasattr(opts, 'length_cost') else 0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr) # Learning rate from core.init

    # Data Loader
    dataset = MiniGridEpisodeDataset(env_factory, opts.episodes_per_epoch, grid_size)
    # Need to fix collate_fn call
    train_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn_minigrid)
    # For validation, if any, a similar dataset/loader would be needed.
    # EGG's EarlyStopperAccuracy expects a validation_data loader.
    # For now, we can skip validation or use a dummy one if EarlyStopper is used.
    # Let's create a small validation set for syntax
    validation_dataset = MiniGridEpisodeDataset(env_factory, opts.batch_size * 2, grid_size) # small validation
    validation_loader = DataLoader(validation_dataset, batch_size=opts.batch_size, collate_fn=collate_fn_minigrid)


    # Trainer
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader, # Pass validation loader
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=True),
            # core.ProgressBar(total=opts.n_epochs, desc="Training EGG"), # Removed due to AttributeError
            EarlyStopperAccuracy(threshold=opts.early_stopping_threshold, validation=True, field_name='reward_mean') 
            # Monitor 'reward_mean' from aux_info
        ]
    )

    print("Starting training...")
    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == "__main__":
    main(sys.argv[1:]) # Pass command-line arguments directly
    print("Training finished or stopped.")
