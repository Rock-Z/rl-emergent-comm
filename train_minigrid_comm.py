import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import sys

import egg.core as core
from egg.core import EarlyStopperAccuracy

from train_minigrid import RNNAgent as NavRNNAgentBase
from train_minigrid import make_env
from compo_vs_generalization.intervention import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoordinateSender(nn.Module):
    """Agent that takes (x,y) coordinates and outputs a hidden state for RNN message generator."""
    def __init__(self, n_coord_features, sender_rnn_hidden_dim):
        super().__init__()
        self.fc = nn.Linear(n_coord_features, sender_rnn_hidden_dim)

    def forward(self, x):
        return self.fc(x)

class NavRNNAgent(NavRNNAgentBase):
    """Modified RNNAgent to process messages and use their content as initial hidden state."""
    def __init__(self, obs_space, num_actions, hidden_size=64, direction_embed_size=64,
                 vocab_size=100, 
                 message_embed_dim=32, 
                 message_rnn_hidden_dim=64, 
                 message_rnn_cell='GRU'):
        super().__init__(obs_space, num_actions, hidden_size, direction_embed_size)
        self.nav_hidden_size = hidden_size

        self.message_embedding = nn.Embedding(vocab_size, message_embed_dim)
        rnn_cell_module = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[message_rnn_cell.upper()]
        self.message_rnn = rnn_cell_module(message_embed_dim, message_rnn_hidden_dim, batch_first=True)
        self.message_to_nav_hidden = nn.Linear(message_rnn_hidden_dim, self.nav_hidden_size)

    def forward_message_processor(self, message_tokens):
        embedded_message = self.message_embedding(message_tokens)
        if isinstance(self.message_rnn, nn.LSTM):
            _, (h_n, _) = self.message_rnn(embedded_message)
        else:
            _, h_n = self.message_rnn(embedded_message)
        final_message_hidden = h_n.squeeze(0) if h_n.dim() == 3 else h_n
        initial_nav_hidden = self.message_to_nav_hidden(final_message_hidden)
        return initial_nav_hidden

    def reset_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.nav_hidden_size, device=device)

    def forward(self, obs_dict, hidden_state):
        img = obs_dict['image'] / 255.0
        img = img.permute(0, 3, 1, 2)
        conv_out = self.conv(img)
        direction_embedded = self.direction_embedding(obs_dict['direction'])
        combined_features = torch.cat((conv_out, direction_embedded), dim=1)
        next_hidden_state = self.rnn(combined_features, hidden_state.clone())
        policy_logits = self.actor(next_hidden_state)
        values = self.critic(next_hidden_state)
        return policy_logits, values.squeeze(-1), next_hidden_state

class NavAgentAsEggReceiver(nn.Module):
    """Wraps NavRNNAgent's message processor for EGG, returning dummy log_probs/entropy."""
    def __init__(self, nav_agent_instance, nav_rnn_hidden_dim, no_comm_mode=False):
        super().__init__()
        self.nav_agent = nav_agent_instance
        self.no_comm_mode = no_comm_mode
        self.nav_rnn_hidden_dim = nav_rnn_hidden_dim

    def forward(self, message_tokens, receiver_input=None, sender_input=None):
        current_device = message_tokens.device
        batch_size = message_tokens.size(0)
        if receiver_input is not None and 'image' in receiver_input and receiver_input['image'] is not None:
             current_device = receiver_input['image'].device
        
        if self.no_comm_mode:
            output = torch.zeros(batch_size, self.nav_rnn_hidden_dim, device=current_device)
        else:
            self.nav_agent.to(current_device)
            output = self.nav_agent.forward_message_processor(message_tokens)
        
        dummy_log_probs = torch.zeros(batch_size, device=output.device)
        dummy_entropy = torch.zeros(batch_size, device=output.device)
        return output, dummy_log_probs, dummy_entropy

class MiniGridNavigationLoss(nn.Module):
    """Custom loss for MiniGrid navigation task within EGG framework."""
    def __init__(self, nav_agent_instance, nav_agent_optimizer, env_factory, 
                 max_episode_steps, gamma, nav_entropy_coeff):
        super().__init__()
        self.nav_agent = nav_agent_instance
        self.nav_agent_optimizer = nav_agent_optimizer
        self.env_factory = env_factory
        self.max_episode_steps = max_episode_steps
        self.gamma = gamma
        self.nav_entropy_coeff = nav_entropy_coeff

    def forward(self, sender_input_coords, _message, _receiver_initial_obs_dict, 
                initial_nav_hidden_state, _labels):
        batch_size = sender_input_coords.size(0)
        cumulative_reward_for_egg_loss = torch.zeros(batch_size, device=sender_input_coords.device)
        batch_episode_rewards = []
        batch_nav_agent_losses = []

        if self.training and torch.is_grad_enabled():
            self.nav_agent_optimizer.zero_grad()

        for i in range(batch_size):
            env_instance = self.env_factory()
            actual_initial_obs_raw, _ = env_instance.reset()
            current_obs_img = torch.FloatTensor(actual_initial_obs_raw['image']).unsqueeze(0).to(device)
            current_obs_dir = torch.tensor([actual_initial_obs_raw['direction']], dtype=torch.long, device=device)
            current_nav_obs_dict = {'image': current_obs_img, 'direction': current_obs_dir}
            current_hidden = initial_nav_hidden_state[i].unsqueeze(0)

            ep_rewards, ep_log_probs, ep_values, ep_entropies, ep_masks = [], [], [], [], []
            for _step in range(self.max_episode_steps):
                policy_logits, value, next_hidden = self.nav_agent(current_nav_obs_dict, current_hidden)
                dist = Categorical(logits=policy_logits)
                action = dist.sample()
                log_prob_action = dist.log_prob(action)
                entropy_action = dist.entropy()
                next_obs_raw_dict, reward, terminated, truncated, _info = env_instance.step(action.item())
                done = terminated or truncated

                ep_rewards.append(reward)
                ep_log_probs.append(log_prob_action)
                ep_values.append(value)
                ep_entropies.append(entropy_action)
                ep_masks.append(1.0 - float(done))

                current_nav_obs_dict = {
                    'image': torch.FloatTensor(next_obs_raw_dict['image']).unsqueeze(0).to(device),
                    'direction': torch.tensor([next_obs_raw_dict['direction']], dtype=torch.long).to(device)
                }
                current_hidden = next_hidden
                if done:
                    break
            env_instance.close()

            returns = []
            R = 0.0
            for r_idx in reversed(range(len(ep_rewards))):
                R = ep_rewards[r_idx] + self.gamma * R * ep_masks[r_idx]
                returns.insert(0, R)
            
            returns_t = torch.tensor(returns, dtype=torch.float, device=device)
            log_probs_t = torch.cat(ep_log_probs)
            values_t = torch.cat(ep_values)
            entropies_t = torch.cat(ep_entropies)
            advantage = returns_t - values_t.detach()
            actor_loss = -(log_probs_t * advantage).mean()
            critic_loss = F.mse_loss(values_t, returns_t)
            entropy_bonus = -self.nav_entropy_coeff * entropies_t.mean()
            nav_agent_loss = actor_loss + 0.5 * critic_loss + entropy_bonus
            
            if self.training and torch.is_grad_enabled():
                nav_agent_loss.backward(retain_graph=True) 
            
            cumulative_reward_for_egg_loss[i] = -torch.tensor(sum(ep_rewards), dtype=torch.float, device=device)
            batch_episode_rewards.append(sum(ep_rewards))
            batch_nav_agent_losses.append(nav_agent_loss.item())

        if self.training and torch.is_grad_enabled():
            self.nav_agent_optimizer.step() 

        final_loss_for_egg_framework = cumulative_reward_for_egg_loss
        aux_info = {
            'acc': torch.tensor(np.mean(batch_episode_rewards), device=device),
            'reward_mean': np.mean(batch_episode_rewards),
            'nav_loss_mean': np.mean(batch_nav_agent_losses),
            'reward_std': np.std(batch_episode_rewards),
        }
        return final_loss_for_egg_framework, aux_info

class MiniGridEpisodeDataset(Dataset):
    """Dataset for MiniGrid episodes, providing initial state and agent position for communication."""
    def __init__(self, env_factory_fn, num_episodes_per_epoch, grid_size=10):
        self.env_factory = env_factory_fn
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.grid_size = grid_size

    def __len__(self):
        return self.num_episodes_per_epoch

    def __getitem__(self, idx):
        env = self.env_factory()
        obs_dict, _info = env.reset()
        agent_pos_x, agent_pos_y = env.agent_pos
        sender_input = torch.tensor(
            [float(agent_pos_x) / self.grid_size, float(agent_pos_y) / self.grid_size],
            dtype=torch.float
        )
        receiver_input_img = torch.FloatTensor(obs_dict['image'])
        receiver_input_dir = torch.tensor(obs_dict['direction'], dtype=torch.long)
        env.close()
        labels = torch.zeros(1)
        return sender_input, labels, {'image': receiver_input_img, 'direction': receiver_input_dir}

def collate_fn_minigrid(batch_list):
    """Collates batch of MiniGrid episodes for DataLoader."""
    sender_inputs, labels, receiver_input_dicts = zip(*batch_list)
    sender_inputs_b = torch.stack(sender_inputs)
    labels_b = torch.stack(labels)
    images_b = torch.stack([d['image'] for d in receiver_input_dicts])
    directions_b = torch.stack([d['direction'] for d in receiver_input_dicts])
    receiver_inputs_b = {'image': images_b, 'direction': directions_b}
    return sender_inputs_b, labels_b, receiver_inputs_b

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sender_cell', type=str, default='gru', help='Sender RNN cell (rnn, gru, lstm)')
    parser.add_argument('--sender_hidden', type=int, default=64, help='Sender RNN hidden size')
    parser.add_argument('--sender_emb', type=int, default=32, help='Sender embedding dimension')
    parser.add_argument('--sender_entropy_coeff', type=float, default=0.01, help='Sender entropy coeff')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.95, help="Early stopping for reward_mean")
    parser.add_argument('--episodes_per_epoch', type=int, default=1000, help="Episodes per epoch")
    parser.add_argument('--length_cost', type=float, default=0.0, help="Cost for message length")

    parser.add_argument('--nav_agent_hidden_size', type=int, default=64, help='Nav agent RNN hidden size')
    parser.add_argument('--nav_agent_lr', type=float, default=2e-5, help='Nav agent learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor for navigation')
    parser.add_argument('--nav_entropy_coeff', type=float, default=0.1, help='Nav agent entropy coeff')
    parser.add_argument('--max_episode_steps', type=int, default=400, help='Max steps per MiniGrid episode')

    parser.add_argument('--nav_message_embed_dim', type=int, default=32, help='Msg embedding dim in NavAgent')
    parser.add_argument('--nav_message_rnn_hidden', type=int, default=64, help='Msg RNN hidden size in NavAgent')
    parser.add_argument('--nav_message_cell', type=str, default='GRU', help='Msg RNN cell in NavAgent')
    
    parser.add_argument('--no_comm', action='store_true', default=False, help='No-communication mode')

    parser.add_argument('--n_attributes', type=int, default=2, help='Number of attributes for Metrics callback (default 2 for x,y coords)')
    parser.add_argument('--n_values', type=int, default=1, help='Number of values for each attribute for Metrics callback (default 1 for coords)')
    parser.add_argument('--stats_freq', type=int, default=1, help='Frequency for logging custom metrics from intervention.Metrics')

    args = core.init(arg_parser=parser, params=params)
    return args

def main(params):
    opts = get_params(params)
    print(f"Using device: {device}")
    if opts.no_comm:
        print("Running in --no-comm mode.")
    else:
        print(f"Running with EGG communication. Options: {opts}")

    def env_factory(): return make_env(video_folder=None)

    sample_env = env_factory()
    obs_space, action_space = sample_env.observation_space, sample_env.action_space
    num_actions, grid_size = action_space.n, sample_env.width
    sample_env.close()

    coord_sender_agent = CoordinateSender(n_coord_features=2, sender_rnn_hidden_dim=opts.sender_hidden).to(device)
    sender = core.RnnSenderReinforce(
        agent=coord_sender_agent, vocab_size=opts.vocab_size, embed_dim=opts.sender_emb,
        hidden_size=opts.sender_hidden, max_len=opts.max_len, force_eos=True, cell=opts.sender_cell.upper()
    ).to(device)

    nav_agent = NavRNNAgent(
        obs_space=obs_space, num_actions=num_actions, hidden_size=opts.nav_agent_hidden_size,
        vocab_size=opts.vocab_size, message_embed_dim=opts.nav_message_embed_dim,
        message_rnn_hidden_dim=opts.nav_message_rnn_hidden, message_rnn_cell=opts.nav_message_cell
    ).to(device)
    nav_agent_optimizer = optim.Adam(nav_agent.parameters(), lr=opts.nav_agent_lr)

    nav_agent_receiver_wrapper = NavAgentAsEggReceiver(
        nav_agent_instance=nav_agent, nav_rnn_hidden_dim=opts.nav_agent_hidden_size,
        no_comm_mode=opts.no_comm
    ).to(device)
    
    minigrid_loss = MiniGridNavigationLoss(
        nav_agent_instance=nav_agent, nav_agent_optimizer=nav_agent_optimizer,
        env_factory=env_factory, max_episode_steps=opts.max_episode_steps,
        gamma=opts.gamma, nav_entropy_coeff=opts.nav_entropy_coeff
    ).to(device)

    game = core.SenderReceiverRnnReinforce(
        sender, nav_agent_receiver_wrapper, minigrid_loss,
        sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=0.0,
        length_cost=opts.length_cost if hasattr(opts, 'length_cost') else 0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    dataset = MiniGridEpisodeDataset(env_factory, opts.episodes_per_epoch, grid_size)
    train_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn_minigrid)
    
    validation_dataset = MiniGridEpisodeDataset(env_factory, opts.batch_size * 2, grid_size)
    validation_loader = DataLoader(validation_dataset, batch_size=opts.batch_size, collate_fn=collate_fn_minigrid)

    metrics_dataset_sender_inputs = []
    if not opts.no_comm:
        for i in range(len(validation_dataset)):
            sender_input, _, _ = validation_dataset[i]
            metrics_dataset_sender_inputs.append(sender_input)
    
    metrics_callback = Metrics(
        dataset=metrics_dataset_sender_inputs, 
        device=opts.device, 
        n_attributes=opts.n_attributes, 
        n_values=opts.n_values, 
        vocab_size=opts.vocab_size, 
        freq=opts.stats_freq
    )

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        EarlyStopperAccuracy(threshold=opts.early_stopping_threshold, validation=True, field_name='reward_mean')
    ]
    if not opts.no_comm and metrics_dataset_sender_inputs:
        callbacks.append(metrics_callback)

    trainer = core.Trainer(
        game=game, optimizer=optimizer, train_data=train_loader, validation_data=validation_loader,
        callbacks=callbacks
    )

    print("Starting EGG training...")
    trainer.train(n_epochs=opts.n_epochs)
    core.close()

if __name__ == "__main__":
    main(sys.argv[1:])
    print("Training finished or stopped.")
