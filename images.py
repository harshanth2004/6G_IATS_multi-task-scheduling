import matplotlib.pyplot as plt
import numpy as np

# Data approximation from base paper (FMPPO vs Baselines)
vehicles = ['5', '10', '15', '20']

# 1. Request Completion Time (Latency) in seconds
fmppo_latency = [180, 250, 380, 580]
ddpg_latency = [280, 420, 480, 750]
sac_latency = [290, 390, 520, 800]
random_latency = [360, 480, 590, 1050]

# 2. Overall Energy Consumption
fmppo_energy = [2000, 3500, 6000, 8000]
ddpg_energy = [3500, 6000, 8500, 10500]
sac_energy = [4000, 5500, 9500, 11500]
random_energy = [6500, 8500, 12000, 14500]

x = np.arange(len(vehicles))
width = 0.2

# --- Plot 1: Latency ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 1.5*width, fmppo_latency, width, label='FMPPO (Ours)', color='#1f77b4')
ax.bar(x - 0.5*width, ddpg_latency, width, label='DDPG', color='#ff7f0e')
ax.bar(x + 0.5*width, sac_latency, width, label='SAC', color='#2ca02c')
ax.bar(x + 1.5*width, random_latency, width, label='Random', color='#d62728')

ax.set_ylabel('Request Completion Time (s)', fontweight='bold')
ax.set_xlabel('Number of Vehicles', fontweight='bold')
ax.set_title('Task Scheduling Latency Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(vehicles)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('latency_chart.png', dpi=300)
plt.show()

# --- Plot 2: Energy ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 1.5*width, fmppo_energy, width, label='FMPPO (Ours)', color='#1f77b4')
ax.bar(x - 0.5*width, ddpg_energy, width, label='DDPG', color='#ff7f0e')
ax.bar(x + 0.5*width, sac_energy, width, label='SAC', color='#2ca02c')
ax.bar(x + 1.5*width, random_energy, width, label='Random', color='#d62728')

ax.set_ylabel('Overall Energy Consumption', fontweight='bold')
ax.set_xlabel('Number of Vehicles', fontweight='bold')
ax.set_title('Energy Efficiency Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(vehicles)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('energy_chart.png', dpi=300)
plt.show()