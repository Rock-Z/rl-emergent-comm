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
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmptyEnvRandom10x10(MiniGridEnv):
    """
    Empty 10x10 grid environment with a random goal location and fixed agent start.
    """
    def __init__(self, size=10, **kwargs):
        # Agent always starts at (1, 1) facing right (direction 0)
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=4 * size * size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent at the fixed starting position
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        # Place a goal square randomly within the grid, excluding walls and agent start
        self.place_obj(Goal(), top=(1,1), size=(width-2, height-2))

        self.mission = "get to the green goal square"

# Register the custom environment
gym.register(
    id="MiniGrid-Empty-Random-10x10-v0",
    entry_point="train_minigrid:EmptyEnvRandom10x10", # Ensure this matches the script name if run directly
)

def make_env(video_folder=None, episode_trigger=None):
    # Initialize environment with rgb_array render_mode for video recording
    env = gym.make("MiniGrid-Empty-Random-10x10-v0", render_mode="rgb_array")
    if video_folder:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger, name_prefix="gameplay")
    #env = RGBImgPartialObsWrapper(env, tile_size=8) # Get RGB image obs
    env = RecordEpisodeStatistics(env) # Records episode statistics
    return env

class RNNAgent(nn.Module):
    def __init__(self, obs_space, num_actions, hidden_size=64, direction_embed_size=8):
        super(RNNAgent, self).__init__()
        
        image_shape = obs_space['image'].shape
        # Direction is encoded as an integer 0-3
        num_directions = 4 
        self.direction_embed_size = direction_embed_size

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_out_size = self._get_conv_output(image_shape)
        
        # Embedding layer for direction
        self.direction_embedding = nn.Embedding(num_directions, direction_embed_size)

        # RNN input size is CNN output + direction embedding
        rnn_input_size = conv_out_size + direction_embed_size
        self.rnn = nn.GRUCell(rnn_input_size, hidden_size)
        
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.hidden = None
        
    def _get_conv_output(self, image_shape):
        # Permute shape to (channels, height, width) for PyTorch
        shape = (image_shape[2], image_shape[0], image_shape[1])
        x = torch.zeros(1, *shape)
        x = self.conv(x)
        return x.size(1)
    
    def reset_hidden(self, batch_size=1):
        self.hidden = torch.zeros(batch_size, 64, device=device)
        
    def forward(self, obs_dict): # Input is now a dictionary
        # Extract image and direction
        img = obs_dict['image'] # Shape: (batch/T, H, W, C) or (T, 1, H, W, C)
        direction = obs_dict['direction'] # Shape: (batch/T,) or (T, 1)

        is_sequence = img.dim() == 5

        if is_sequence:
            T = img.shape[0]
            img = img.squeeze(1) # Squeeze batch dim: (T, H, W, C)
            direction = direction.squeeze(1) # Squeeze batch dim: (T,)
        else:
            T = 1 # Treat as sequence of length 1

        # --- Process Image --- 
        img = img / 255.0
        img = img.permute(0, 3, 1, 2) # (T or batch, C, H, W)
        conv_out = self.conv(img) # (T or batch, conv_out_size)

        # --- Process Direction --- 
        direction_embedded = self.direction_embedding(direction) # (T or batch, direction_embed_size)

        # --- Combine Features --- 
        combined_features = torch.cat((conv_out, direction_embedded), dim=1) # (T or batch, rnn_input_size)

        # --- Process with RNN --- 
        if is_sequence:
            hidden_state = torch.zeros(1, 64, device=device)
            outputs = []
            for t in range(T):
                rnn_input = combined_features[t].unsqueeze(0)
                hidden_state = self.rnn(rnn_input, hidden_state)
                outputs.append(hidden_state)
            rnn_output = torch.stack(outputs).squeeze(1)
        else:
            batch_size = combined_features.size(0)
            if self.hidden is None or self.hidden.size(0) != batch_size:
                self.reset_hidden(batch_size)
            self.hidden = self.rnn(combined_features, self.hidden)
            rnn_output = self.hidden

        policy_logits = self.actor(rnn_output)
        values = self.critic(rnn_output)

        policy = F.softmax(policy_logits, dim=-1)
        return policy, values.squeeze(-1)

def compute_returns(rewards, masks, gamma=0.99):
    returns = []
    R = 0
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(agent, optimizer, states_dicts, actions, returns, values, entropy_coef=0.01):
    # Process the list of state dictionaries
    images = torch.stack([torch.FloatTensor(s['image']).to(device) for s in states_dicts])
    directions = torch.tensor([s['direction'] for s in states_dicts], dtype=torch.long, device=device)
    
    # Create input dictionary for the agent
    obs_input = {'image': images, 'direction': directions}

    actions = torch.tensor(actions, dtype=torch.long, device=device)
    returns = torch.tensor(returns, dtype=torch.float, device=device)
    values = torch.tensor(values, dtype=torch.float, device=device).squeeze()
    
    # Note: agent expects dictionary input now
    logits, _ = agent(obs_input)
    dist = Categorical(logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    advantage = returns - values
    
    policy_loss = -(log_probs * advantage.detach()).mean()
    value_loss = F.mse_loss(values.view(-1), returns.view(-1))
    
    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def run_recorded_episode(agent, video_folder, episode_num):
    print(f"Recording episode {episode_num}...")
    rec_env = make_env(video_folder=video_folder, episode_trigger=lambda x: x == 0)
    
    state_dict, _ = rec_env.reset()
    agent.reset_hidden()
    done = False
    episode_reward = 0.0

    while not done:
        # Prepare dictionary input for agent
        img_tensor = torch.FloatTensor(state_dict['image']).unsqueeze(0).to(device)
        dir_tensor = torch.tensor([state_dict['direction']], dtype=torch.long, device=device)
        obs_input = {'image': img_tensor, 'direction': dir_tensor}

        with torch.no_grad():
            # Agent takes dictionary input
            policy, _ = agent(obs_input)
            dist = Categorical(policy)
            action = dist.sample()
        
        next_state_dict, reward, terminated, truncated, _ = rec_env.step(action.item())
        done = terminated or truncated
        state_dict = next_state_dict
        episode_reward += reward

    rec_env.close()
    print(f"Finished recording episode {episode_num}. Reward: {episode_reward:.2f}")

def main():
    parser = argparse.ArgumentParser(description="RNN Policy Gradient for MiniGrid")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint interval")
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    env = make_env()
    
    # Pass the observation space dictionary to the agent
    obs_space = env.observation_space
    num_actions = env.action_space.n
    agent = RNNAgent(obs_space, num_actions).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(1, args.num_episodes + 1):
        state_dict, _ = env.reset() # Now returns a dictionary
        agent.reset_hidden()
        
        done = False
        episode_reward = 0.0
        
        states_dicts = [] # Store state dictionaries
        actions = []
        rewards = []
        values = []
        masks = []
        
        while not done:
            # Prepare dictionary input for agent
            img_tensor = torch.FloatTensor(state_dict['image']).unsqueeze(0).to(device)
            dir_tensor = torch.tensor([state_dict['direction']], dtype=torch.long, device=device)
            obs_input = {'image': img_tensor, 'direction': dir_tensor}

            with torch.no_grad():
                # Agent takes dictionary input
                policy, value = agent(obs_input)
                dist = Categorical(policy)
                action = dist.sample()
            
            next_state_dict, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store the full state dictionary
            states_dicts.append(state_dict)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            masks.append(1 - done)
            
            state_dict = next_state_dict
            episode_reward += reward
            total_steps += 1
        
        returns = compute_returns(rewards, masks, gamma=args.gamma)
        
        # Pass list of state dictionaries to train function
        loss = train(agent, optimizer, states_dicts, actions, returns, values, entropy_coef=args.entropy_coef)
        
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