import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers import RecordVideo
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(video_folder=None, episode_trigger=None):
    # Initialize environment with rgb_array render_mode for video recording
    env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="rgb_array")
    if video_folder:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix="gameplay")
    env = RGBImgPartialObsWrapper(env, tile_size=8) # Get RGB image obs
    env = ImgObsWrapper(env) # Get rid of the Dict observation space
    env = RecordEpisodeStatistics(env) # Records episode statistics
    return env

class RNNAgent(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=64):
        super(RNNAgent, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        conv_out_size = self._get_conv_output(input_shape)
        
        self.rnn = nn.GRUCell(conv_out_size, hidden_size)
        
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.hidden = None
        
    def _get_conv_output(self, shape):
        # Permute shape to (channels, height, width) for PyTorch
        shape = (shape[2], shape[0], shape[1])
        x = torch.zeros(1, *shape)
        x = self.conv(x)
        return x.size(1)
    
    def reset_hidden(self, batch_size=1):
        self.hidden = torch.zeros(batch_size, 64, device=device)
        
    def forward(self, x): # Input shape: (batch, H, W, C) or (T, 1, H, W, C)
        is_sequence = x.dim() == 5

        if is_sequence:
            T = x.shape[0]
            x = x.squeeze(1) # Squeeze batch dim: (T, H, W, C)
        else:
            T = 1 # Treat as sequence of length 1

        # Normalize and permute for CNN: (T or batch, C, H, W)
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)

        conv_out = self.conv(x)

        # Process sequence with RNN
        if is_sequence:
            # Training phase: process sequence step-by-step
            hidden_state = torch.zeros(1, 64, device=device) # Local hidden state for sequence
            outputs = []
            for t in range(T):
                rnn_input = conv_out[t].unsqueeze(0)
                hidden_state = self.rnn(rnn_input, hidden_state)
                outputs.append(hidden_state)
            rnn_output = torch.stack(outputs).squeeze(1) # (T, hidden_size)
        else:
            # Rollout phase: use persistent hidden state
            batch_size = conv_out.size(0)
            if self.hidden is None or self.hidden.size(0) != batch_size:
                self.reset_hidden(batch_size)
            self.hidden = self.rnn(conv_out, self.hidden)
            rnn_output = self.hidden # (1, hidden_size)

        policy_logits = self.actor(rnn_output)
        values = self.critic(rnn_output)

        policy = F.softmax(policy_logits, dim=-1)

        # Return policy probs and state values
        return policy, values.squeeze(-1)

def compute_returns(rewards, masks, gamma=0.99):
    returns = []
    R = 0
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(agent, optimizer, states, actions, returns, values, entropy_coef=0.01):
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    returns = torch.tensor(returns, dtype=torch.float, device=device)
    values = torch.tensor(values, dtype=torch.float, device=device).squeeze()
    
    logits, _ = agent(states)
    dist = Categorical(logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    advantage = returns - values
    
    policy_loss = -(log_probs * advantage.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    
    # Total loss: policy gradient + value loss - entropy bonus
    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def run_recorded_episode(agent, video_folder, episode_num):
    print(f"Recording episode {episode_num}...")
    # Create env with video recording enabled for this specific episode
    rec_env = make_env(video_folder=video_folder, episode_trigger=lambda x: x == 0) # Record only the first episode (index 0)
    
    state, _ = rec_env.reset()
    agent.reset_hidden() # Reset agent's hidden state for the recording
    done = False
    episode_reward = 0.0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, _ = agent(state_tensor)
            dist = Categorical(policy)
            action = dist.sample()
        
        next_state, reward, terminated, truncated, _ = rec_env.step(action.item())
        done = terminated or truncated
        state = next_state
        episode_reward += reward

    rec_env.close()
    print(f"Finished recording episode {episode_num}. Reward: {episode_reward:.2f}")

def main():
    parser = argparse.ArgumentParser(description="RNN Policy Gradient for MiniGrid")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint interval")
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    env = make_env()
    
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = RNNAgent(input_shape, num_actions).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(1, args.num_episodes + 1):
        state, _ = env.reset()
        agent.reset_hidden()
        
        done = False
        episode_reward = 0.0
        
        # Store trajectory data for this episode
        states = []
        actions = []
        rewards = []
        values = []
        masks = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                policy, value = agent(state_tensor)
                dist = Categorical(policy)
                action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state_tensor)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            masks.append(1 - done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
        
        returns = compute_returns(rewards, masks, gamma=args.gamma)
        
        loss = train(agent, optimizer, states, actions, returns, values, entropy_coef=args.entropy_coef)
        
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{args.num_episodes}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
        
        if episode % args.checkpoint_interval == 0 or episode == args.num_episodes:
            # Record a gameplay video
            video_path = os.path.join("videos", f"episode_{episode}")
            run_recorded_episode(agent, video_path, episode)

            # Save checkpoint
            checkpoint_path = os.path.join("checkpoints", f"rnn_agent_episode_{episode}.pt")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': np.mean(episode_rewards[-10:]),
                'total_steps': total_steps,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    main()