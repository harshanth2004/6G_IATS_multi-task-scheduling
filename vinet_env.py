import numpy as np

class VINETEnv:
    def __init__(self, num_vehicles=10):
        # 1. Highway & Network Settings [cite: 501, 502, 503]
        self.highway_len = 13000.0  # 13km highway
        self.num_rsus = 8
        self.rsu_positions = np.linspace(0, self.highway_len, self.num_rsus)
        self.num_vehicles = num_vehicles
        
        # 2. Computation & Communication Parameters [cite: 462, 464, 468, 472, 474]
        self.B = 100e6  # 100 MHz Bandwidth
        self.p_v_dbm = 1.0  # Vehicle transmission power in dBm
        self.p_v = (10 ** (self.p_v_dbm / 10)) * 1e-3  # Convert dBm to Watts
        self.noise_power_dbm = -174.0 # White Gaussian noise
        self.sigma2 = (10 ** (self.noise_power_dbm / 10)) * 1e-3 * self.B # Noise in Watts
        
        self.f_v = 1.0  # Vehicle CPU power (scaled)
        self.f_s = 2.0  # Edge Server CPU power (scaled)
        self.xi = 1e-11 # Power consumption coefficient [cite: 247]
        self.gamma = 2  # Power consumption coefficient [cite: 247]
        
        self.max_energy_history = 100.0 # Used for reward normalization
        
        self.reset()

    def reset(self):
        # Initialize vehicle positions and Poisson-distributed speeds [cite: 503]
        self.v_positions = np.random.uniform(0, self.highway_len, self.num_vehicles)
        self.v_speeds = np.random.poisson(25, self.num_vehicles) # Avg 25 m/s
        
        # Queues: [Incoming Queue, Processing Queue] for each vehicle
        self.queues = np.zeros((self.num_vehicles, 2))
        
        return self._get_obs(0)

    def _get_obs(self, vehicle_idx):
        v_pos = self.v_positions[vehicle_idx]
        nearest_rsu = self.rsu_positions[np.argmin(np.abs(self.rsu_positions - v_pos))]
        
        # Task Generation: 2-20 CPU cycles, 2-20 Mb data size [cite: 461, 466]
        task_cpu = np.random.uniform(2.0, 20.0) 
        task_size = np.random.uniform(2.0, 20.0)
        
        # State Vector [cite: 350, 352]
        state = np.array([
            v_pos / self.highway_len, 
            nearest_rsu / self.highway_len,
            task_cpu / 20.0, 
            task_size / 20.0,
            self.queues[vehicle_idx][0] / 50.0, # Normalized Incoming Queue
            self.queues[vehicle_idx][1] / 50.0  # Normalized Processing Queue
        ], dtype=np.float32)
        
        return state

    def _calculate_data_rate(self, distance):
        # Prevent division by zero if distance is 0
        distance = max(distance, 1.0)
        
        # Simplified Path Loss (Free space)
        path_loss = distance ** 2 
        
        # SNR Calculation: Eq (3) [cite: 218]
        snr = self.p_v / (path_loss * self.sigma2)
        
        # Data Rate Calculation: Eq (4) & (5) [cite: 224, 228]
        # Result in Mbps
        rate = self.B * np.log2(1 + snr) / 1e6 
        return max(rate, 0.1) # Ensure a minimum data rate

    def _find_best_neighbor(self, v_idx):
        # Target Selection for V2V [cite: 332]
        best_score = -1
        best_neighbor = v_idx
        
        for j in range(self.num_vehicles):
            if j == v_idx: continue
            
            dist = np.abs(self.v_positions[v_idx] - self.v_positions[j])
            if dist > 500: continue # Communication range limit
            
            workload = self.queues[j][1]
            # Score = (1/distance) * (1 - workload/max_workload)
            score = (1.0 / max(dist, 1.0)) * (1.0 - min(workload / 50.0, 1.0))
            
            if score > best_score:
                best_score = score
                best_neighbor = j
                
        return best_neighbor, np.abs(self.v_positions[v_idx] - self.v_positions[best_neighbor])

    def step(self, vehicle_idx, action):
        v_pos = self.v_positions[vehicle_idx]
        task_cpu = np.random.uniform(2.0, 20.0)
        task_size = np.random.uniform(2.0, 20.0)
        
        latency = 0.0
        energy = 0.0
        
        # ACTION 0: V2V (Offload to Neighbor Vehicle) [cite: 365]
        if action == 0:
            target_idx, dist = self._find_best_neighbor(vehicle_idx)
            rate = self._calculate_data_rate(dist)
            
            t_trans = task_size / rate
            t_exe = task_cpu / self.f_v
            latency = t_trans + t_exe + self.queues[target_idx][0] # Eq (8) [cite: 256]
            
            # Energy: Transmission + Neighbor's execution [cite: 274]
            energy = (self.p_v * t_trans) + (self.xi * (self.f_v ** self.gamma) * task_cpu)
            
            # Add to neighbor's queue
            self.queues[target_idx][0] += task_size
            
        # ACTION 1: V2I (Offload to RSU) [cite: 366]
        elif action == 1:
            nearest_rsu = self.rsu_positions[np.argmin(np.abs(self.rsu_positions - v_pos))]
            dist = np.abs(v_pos - nearest_rsu)
            rate = self._calculate_data_rate(dist)
            
            t_trans = task_size / rate
            t_exe = task_cpu / self.f_s
            latency = t_trans + t_exe # Eq (8) [cite: 256]
            
            # Energy: Only transmission (server energy isn't drawn from vehicle) [cite: 274]
            energy = self.p_v * t_trans
            
        # ACTION 2: Local Execution [cite: 367]
        elif action == 2:
            latency = task_cpu / self.f_v + self.queues[vehicle_idx][1] # Eq (6) [cite: 243]
            energy = self.xi * (self.f_v ** self.gamma) * task_cpu # Eq (7) [cite: 243]
            self.queues[vehicle_idx][1] += task_cpu
            
        # Update max energy history
        self.max_energy_history = max(self.max_energy_history, energy + 1e-5)
        
        # REWARD CALCULATION: Multi-objective (Time and Energy) Eq (16) [cite: 370]
        # Note: We subtract these because we want to MINIMIZE latency and energy.
        alpha, beta = 0.5, 0.5
        reward = -(alpha * latency) - (beta * (energy / self.max_energy_history))
        
        # Update Highway Physics
        self.v_positions += self.v_speeds
        self.v_positions %= self.highway_len
        
        # Cool down queues slightly every step
        self.queues = np.maximum(self.queues - 0.5, 0)
        
        next_state = self._get_obs(vehicle_idx)
        done = False # Continuous environment
        
        return next_state, reward, done