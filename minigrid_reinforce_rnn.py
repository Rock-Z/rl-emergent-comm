import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64):
        super(RNNPolicy, self).__init__()
        
        # Input size depends on observation space
        if isinstance(observation_space, gym.spaces.Box):
            # For image observations (H x W x C)
            h, w, c = observation_space.shape
            self.image_conv = nn.Sequential(
                nn.Conv2d(c, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
            
            # Compute the size of the flattened output after convolutions
            test_tensor = torch.zeros(1, c, h, w)
            with torch.no_grad():
                test_output = self.image_conv(test_tensor)
                conv_output_size = test_output.numel()
            
            self.fc = nn.Linear(conv_output_size, hidden_size)
        else:
            # For other observation spaces
            raise NotImplementedError("Unsupported observation space")
        
        # RNN layer
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        
        # Action head
        self.action_head = nn.Linear(hidden_size, action_space.n)
        
        # Value head for critic
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initial hidden state
        self.hidden = None
        
        # Store for saving
        self.hidden_size = hidden_size
        
    def forward(self, x, hidden=None):
        # Preprocess image input
        if len(x.shape) == 4:  # [batch, height, width, channels]
            x = x.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        elif len(x.shape) == 3:  # [height, width, channels]
            x = x.permute(2, 0, 1).unsqueeze(0)  # [1, channels, height, width]
        
        x = self.image_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        
        # Use provided hidden state or initialize a new one
        if hidden is None:
            if self.hidden is None:
                hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            else:
                hidden = self.hidden
        
        # Update hidden state
        hidden = self.rnn(x, hidden)
        self.hidden = hidden.detach()  # Detach to prevent backprop through episodes
        
        # Action probabilities
        action_probs = F.softmax(self.action_head(hidden), dim=-1)
        
        # State value
        state_value = self.value_head(hidden)
        
        return action_probs, state_value, hidden
    
    def reset_hidden(self, batch_size=1):
        self.hidden = torch.zeros(batch_size, self.hidden_size, device=device)

def select_action(policy, state, batch=False):
    """Select an action from policy given state"""
    state = torch.FloatTensor(state).to(device)
    
    # Get action probabilities and value
    probs, value, _ = policy(state)
    
    # Create a distribution and sample
    m = Categorical(probs)
    action = m.sample()
    
    # When in batch mode, return tensors directly
    if batch:
        return action, m.log_prob(action), value
    
    # Otherwise return scalar values for single environment case
    return action.item(), m.log_prob(action), value

def update_policy(policy, optimizer, rewards, log_probs, values, gamma=0.99, eps=1e-8):
    """Update policy parameters using REINFORCE with baseline"""
    # Calculate returns
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    
    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    
    # Calculate policy loss
    policy_loss = []
    value_loss = []
    
    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device).view(1, 1)))
    
    # Combine losses
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    
    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def make_env_fn(env_id, seed, render_mode=None, fully_observed=False):
    """Create a callable environment function for vectorized environments"""
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed)
        
        # Optionally use fully observed wrapper
        if fully_observed:
            env = RGBImgObsWrapper(env)
        else:
            env = RGBImgPartialObsWrapper(env)
        
        # Always use image observations
        env = ImgObsWrapper(env)
        return env
    return _init

def make_vec_env(env_id="MiniGrid-FourRooms-v0", num_envs=1, seed=42, 
                render_mode=None, fully_observed=False, asynchronous=True):
    """Create a vectorized environment with multiple parallel environments"""
    env_fns = [make_env_fn(env_id, seed + i, render_mode, fully_observed) 
               for i in range(num_envs)]
    
    if asynchronous:
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)

def make_env(env_id="MiniGrid-FourRooms-v0", seed=42, render_mode=None, fully_observed=False):
    """Create and wrap the environment"""
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    
    # Optionally use fully observed wrapper
    if fully_observed:
        env = RGBImgObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)
    
    # Always use image observations
    env = ImgObsWrapper(env)
    
    return env

def save_frames_as_gif(frames, path='./videos/', filename='four_rooms_reinforce.gif'):
    """Save frames as gif"""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    
    # Create figure and animation
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        return [patch]
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filepath, writer='imagemagick', fps=10)
    plt.close()
    print(f"GIF saved to {filepath}")
    return filepath

def evaluate_policy(env, policy, num_episodes=1, render=False, save_gif=False):
    """Evaluate policy for a number of episodes and optionally render and save as gif"""
    total_rewards = []
    all_frames = []
    
    for episode in range(num_episodes):
        policy.reset_hidden()
        state, _ = env.reset()
        episode_reward = 0
        frames = []
        done = False
        truncated = False
        
        while not (done or truncated):
            if render:
                frames.append(env.render())
            
            action, _, _ = select_action(policy, state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        
        if save_gif and frames:
            all_frames.extend(frames)
    
    if save_gif and all_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_frames_as_gif(all_frames, filename=f'four_rooms_reinforce_{timestamp}.gif')
    
    return np.mean(total_rewards)

def train_reinforce(env_id="MiniGrid-FourRooms-v0", num_episodes=1000, hidden_size=64, 
                    lr=0.001, gamma=0.99, eval_interval=100, save_interval=500, 
                    fully_observed=False, render_eval=True, save_model=True, batch_size=1):
    """Train agent using REINFORCE algorithm"""
    
    if batch_size > 1:
        # Create vectorized environment for batch training
        env = make_vec_env(
            env_id=env_id, 
            num_envs=batch_size, 
            seed=42, 
            fully_observed=fully_observed
        )
        # Single evaluation environment
        eval_env = make_env(env_id=env_id, render_mode='rgb_array', fully_observed=fully_observed)
        
        # Initialize policy for batch training
        single_env = make_env(env_id=env_id)  # Temp env to get spaces
        policy = RNNPolicy(single_env.observation_space, single_env.action_space, hidden_size=hidden_size).to(device)
        single_env.close()
    else:
        # Create regular environment for single instance training
        env = make_env(env_id=env_id, fully_observed=fully_observed)
        eval_env = make_env(env_id=env_id, render_mode='rgb_array', fully_observed=fully_observed)
        
        # Initialize policy
        policy = RNNPolicy(env.observation_space, env.action_space, hidden_size=hidden_size).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    rewards_log = []
    best_reward = -float('inf')
    
    print(f"Starting training on {env_id} for {num_episodes} episodes with batch size {batch_size}...")
    
    if batch_size > 1:
        # Batch training loop
        states, _ = env.reset()
        
        for episode in range(1, num_episodes + 1):
            # Reset hidden states for all environments
            policy.reset_hidden(batch_size=batch_size)
            
            # Storage for batch episodes
            batch_rewards = [[] for _ in range(batch_size)]
            batch_log_probs = [[] for _ in range(batch_size)]
            batch_values = [[] for _ in range(batch_size)]
            
            # Track which environments are done
            dones = [False] * batch_size
            all_done = False
            
            while not all_done:
                # Select batch actions
                actions, log_probs, values = select_action(policy, states, batch=True)
                
                # Step environments
                next_states, rewards, terminations, truncations, _ = env.step(actions.cpu().numpy())
                
                # Store experience
                for i in range(batch_size):
                    if not dones[i]:
                        batch_rewards[i].append(rewards[i])
                        batch_log_probs[i].append(log_probs[i])
                        batch_values[i].append(values[i])
                        
                        # Check if this environment is done
                        dones[i] = terminations[i] or truncations[i]
                
                # Update states
                states = next_states
                
                # Check if all environments are done
                all_done = all(dones)
                
                # Reset finished environments if needed
                if not all_done:
                    for i in range(batch_size):
                        if dones[i]:
                            # Reset just this environment
                            reset_states, _ = env.reset()
                            # We reset only the environments that are done, without affecting others
                            # Note that in vectorized environments, we need to manage this carefully
                            states[i] = reset_states[i]
            
            # Update policy for each environment's trajectory
            total_loss = 0
            for i in range(batch_size):
                loss = update_policy(
                    policy, optimizer, batch_rewards[i], batch_log_probs[i], batch_values[i], gamma
                )
                total_loss += loss
                rewards_log.append(sum(batch_rewards[i]))
            
            avg_loss = total_loss / batch_size
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_log[-batch_size*10:])
                print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
            
            # Evaluate policy
            if episode % eval_interval == 0 or episode == num_episodes:
                eval_reward = evaluate_policy(
                    eval_env, policy, num_episodes=5, render=render_eval, 
                    save_gif=(episode % save_interval == 0 or episode == num_episodes)
                )
                print(f"Evaluation at episode {episode}: Avg Reward: {eval_reward:.2f}")
                
                # Save best model
                if save_model and eval_reward > best_reward:
                    best_reward = eval_reward
                    save_path = f"./checkpoints/four_rooms_reinforce_best.pt"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'episode': episode,
                        'reward': best_reward,
                        'hidden_size': hidden_size,
                        'batch_size': batch_size
                    }, save_path)
                    print(f"Model saved to {save_path}")
            
            # Reset environments for next episode
            states, _ = env.reset()
    
    else:
        # Original single-environment training loop
        for episode in range(1, num_episodes + 1):
            # Reset environment and agent state
            policy.reset_hidden()
            state, _ = env.reset()
            
            # Collect experience for one episode
            rewards = []
            log_probs = []
            values = []
            done = False
            truncated = False
            
            while not (done or truncated):
                action, log_prob, value = select_action(policy, state)
                state, reward, done, truncated, _ = env.step(action)
                
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
            
            # Update policy
            loss = update_policy(policy, optimizer, rewards, log_probs, values, gamma)
            
            # Log episode result
            episode_reward = sum(rewards)
            rewards_log.append(episode_reward)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_log[-10:])
                print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
            
            # Evaluate policy
            if episode % eval_interval == 0 or episode == num_episodes:
                eval_reward = evaluate_policy(
                    eval_env, policy, num_episodes=5, render=render_eval, 
                    save_gif=(episode % save_interval == 0 or episode == num_episodes)
                )
                print(f"Evaluation at episode {episode}: Avg Reward: {eval_reward:.2f}")
                
                # Save best model
                if save_model and eval_reward > best_reward:
                    best_reward = eval_reward
                    save_path = f"./checkpoints/four_rooms_reinforce_best.pt"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'episode': episode,
                        'reward': best_reward,
                        'hidden_size': hidden_size,
                        'batch_size': batch_size
                    }, save_path)
                    print(f"Model saved to {save_path}")
    
    # Final save
    if save_model:
        save_path = f"./checkpoints/four_rooms_reinforce_final.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': num_episodes,
            'reward': eval_reward,
            'hidden_size': hidden_size,
            'batch_size': batch_size
        }, save_path)
        print(f"Final model saved to {save_path}")
    
    # Generate final gif
    evaluate_policy(eval_env, policy, num_episodes=1, render=True, save_gif=True)
    
    env.close()
    eval_env.close()
    
    return policy, rewards_log

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train RNN agent with REINFORCE on Four Rooms')
    parser.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0', help='Environment ID')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size for RNN')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=50, help='Save GIF interval')
    parser.add_argument('--fully-observed', action='store_true', help='Use fully observed wrapper')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering during eval')
    parser.add_argument('--no-save', action='store_true', help='Disable model saving')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for training')
    args = parser.parse_args()
    
    # Train the agent
    policy, rewards = train_reinforce(
        env_id=args.env,
        num_episodes=args.episodes,
        hidden_size=args.hidden_size,
        lr=args.learning_rate,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        fully_observed=args.fully_observed,
        render_eval=not args.no_render,
        save_model=not args.no_save,
        batch_size=args.batch_size
    )
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('./figures/', exist_ok=True)
    plt.savefig('./figures/training_progress.png')
    print("Training progress plot saved to ./figures/training_progress.png")

if __name__ == "__main__":
    main()