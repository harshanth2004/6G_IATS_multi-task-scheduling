import torch
import numpy as np
from vinet_env import VINETEnv
from fmppo_agent import PPOAgent

def evaluate():
    # 1. Initialize the environment and agent architecture
    env = VINETEnv()
    agent = PPOAgent(state_dim=6, action_dim=3)
    
    model_path = 'trained_models/fmppo_model.pth'
    
    # 2. Load the trained weights into the Actor-Critic networks
    try:
        agent.policy.load_state_dict(torch.load(model_path))
        print(f"✅ Successfully loaded trained weights from {model_path}")
    except FileNotFoundError:
        print(f"❌ Could not find {model_path}. Make sure you run main.py to save the model first.")
        return

    # 3. Set the network to Evaluation Mode (disables training dropout/exploration)
    agent.policy.eval()
    
    test_episodes = 5
    steps_per_episode = 2048

    print("-" * 40)
    print("🚀 STARTING FMPPO EVALUATION")
    print("-" * 40)

    for i_episode in range(1, test_episodes + 1):
        state = env.reset()
        ep_reward = 0
        
        for t in range(steps_per_episode):
            # 4. Greedy Action Selection (No Exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                # Get the probabilities from the trained Actor network
                action_probs = agent.policy.actor(state_tensor)
                # Pick the action with the highest probability
                action = torch.argmax(action_probs).item()
            
            # 5. Execute the optimal action
            next_state, reward, done = env.step(0, action)
            
            # Background vehicles still create the dynamic 6G environment
            for v_idx in range(1, env.num_vehicles):
                random_action = np.random.choice([0, 1, 2])
                env.step(v_idx, random_action)
                
            state = next_state
            ep_reward += reward
            
            if done:
                break
                
        # Calculate the average reward to compare with your training graph
        avg_reward = ep_reward / steps_per_episode
        print(f"📊 Evaluation Episode {i_episode}/{test_episodes} \t Average Reward: {avg_reward:.4f}")

if __name__ == '__main__':
    evaluate()