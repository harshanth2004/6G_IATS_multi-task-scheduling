import torch
import matplotlib.pyplot as plt
import os
import datetime
from vinet_env import VINETEnv
from fmppo_agent import PPOAgent
from tqdm import tqdm

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

def train():
    env = VINETEnv()
    # state_dim=6, action_dim=3 based on our previous setup
    agent = PPOAgent(state_dim=6, action_dim=3)
    memory = Memory()
    
    episodes = 500
    steps_per_episode = 2048 # Corrected to match Table III 
    
    # List to store rewards for the graph
    reward_history = []

    for i_episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        ep_reward = 0
        
        for t in range(steps_per_episode):
            # 1. Learning Agent (Vehicle 0) selects action based on policy [cite: 304]
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(0, action)
            
            # 2. Other vehicles act randomly to create the dynamic environment [cite: 304-306]
            for v_idx in range(1, env.num_vehicles):
                random_action = torch.randint(0, 3, (1,)).item()
                env.step(v_idx, random_action)
            
            # 3. Store data for learning
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            
            state = next_state
            ep_reward += reward
            
            if done: 
                break
                
        # 4. Update agent at the end of the 2048-step episode
        agent.update(memory)
        memory = Memory() # Clear memory for the next episode
        
        # Track average reward per step for the plot
        avg_ep_reward = ep_reward / steps_per_episode
        reward_history.append(avg_ep_reward)
            
        if i_episode % 10 == 0:
            print(f" Episode {i_episode} \t Avg Reward: {avg_ep_reward:.4f}")

    # 5. Generate the Convergence Graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, episodes + 1), reward_history, label='FMPPO Learning Vehicle')
    plt.title('Reward Convergence over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
   # Create a unique timestamp (e.g., "20260227_170530")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- SAVE PLOT IN A SEPARATE FOLDER ---
    os.makedirs('plots', exist_ok=True) # Creates the 'plots' folder
    plot_filename = f"convergence_plot_{timestamp}.png"
    plot_path = os.path.join('plots', plot_filename)
    
    plt.savefig(plot_path)
    print(f"Training complete! Saved convergence plot to '{plot_path}'")
    
    # 6. Save the trained model into a specific folder with matching timestamp
    os.makedirs('trained_models', exist_ok=True) # Creates the 'trained_models' folder
    model_filename = f"fmppo_model_{timestamp}.pth"
    model_path = os.path.join('trained_models', model_filename)
    
    torch.save(agent.policy.state_dict(), model_path)
    print(f"Model weights successfully saved to '{model_path}'!")

if __name__ == '__main__':
    train()