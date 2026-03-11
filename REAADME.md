# Reinforcement Learning Based Edge-End Collaboration for Multi-task Scheduling in 6G Intelligent Autonomous Transport Systems

**Institution:** Netaji Subhas University of Technology (NSUT), Delhi  
**Supervisors:** Dr. Rashmi Chaudhry & Ms. Surabhi Purwar  
**Team Members:** Harshanth Raja (2022UCS1631), Ankit (2022UCS1656), Rajnish Meena (2022UCS1556)  

## Project Overview
This repository contains the simulation environment and Deep Reinforcement Learning (DRL) agent for our Bachelor Thesis Project. We propose a Fully-decentralized Multi-agent Proximal Policy Optimization (FMPPO) algorithm to act as a real-time runtime optimizer for 6G Vehicle-Infrastructure Networks (VINET). 

The goal of the agent is to jointly minimize the expected request completion time (latency) and the overall energy consumption of autonomous vehicles by intelligently scheduling tasks across:
1. **Local Execution** (Onboard CPU)
2. **V2I Offloading** (Edge Servers / RSUs)
3. **V2V Offloading** (Neighboring Idle Vehicles)

## Repository Structure
* `environment/`: Contains the custom 6G VINET simulation environment (mobility, latency, and energy models).
* `agent/`: Contains the Actor-Critic neural network architecture and PPO update mechanics.
* `utils/`: Helper functions for math and logging.
* `evaluate.py`: Script to test the trained model against baseline heuristics (Local-Only, Edge-Only, Random).

## Setup & Installation

**1. Clone the repository**
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

**2. Create a Virtual Environment**
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

**3. Install Dependencies**
pip install -r requirements.txt

## How to Run
To begin training the FMPPO agent from scratch, run:
python train.py

To evaluate a pre-trained model and generate performance graphs, run:
python evaluate.py