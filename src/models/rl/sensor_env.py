#RL environment for dynamic sensor selection
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

class SensorSelectionEnv(gym.Env):
    """
    Custom OpenAI Gym environment for RL-based dynamic sensor selection
    in the AeroTurb-RL system for aircraft turbulence detection
    """
    def _init_(self, 
                 num_sensors=50, 
                 max_selections=10, 
                 flight_state_dim=20,
                 max_episode_steps=100,
                 energy_budget=100.0):
        super()._init_()
        
        # Environment parameters
        self.num_sensors = num_sensors
        self.max_selections = max_selections
        self.flight_state_dim = flight_state_dim
        self.max_episode_steps = max_episode_steps
        self.energy_budget = energy_budget
        
        # Action space: binary selection of sensors
        self.action_space = spaces.MultiBinary(num_sensors)
        
        # Observation space: flight state + sensor availability + energy levels + time
        obs_dim = flight_state_dim + num_sensors + num_sensors + 1  # state + availability + energy + time
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Sensor characteristics (randomized but consistent across episodes)
        self.sensor_energy_cost = np.random.uniform(0.1, 2.0, num_sensors)
        self.sensor_informativeness = np.random.uniform(0.3, 1.0, num_sensors)
        self.sensor_reliability = np.random.uniform(0.7, 1.0, num_sensors)
        
        # Flight phase mapping (affects turbulence probability and sensor effectiveness)
        self.flight_phases = ['taxi', 'takeoff', 'climb', 'cruise', 'descent', 'approach', 'landing']
        
        # Initialize episode variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.energy_remaining = self.energy_budget
        self.total_reward = 0.0
        
        # Initialize flight state
        self.flight_phase = self._sample_flight_phase()
        self.flight_state = self._generate_flight_state()
        self.turbulence_present = self._generate_turbulence_scenario()
        
        # All sensors initially available
        self.sensor_availability = np.ones(self.num_sensors)
        
        # Sensor degradation over time (simulating wear/environmental effects)
        self.sensor_degradation = np.zeros(self.num_sensors)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one step of the environment"""
        # Validate action
        action = np.array(action, dtype=int)
        selected_sensors = np.where(action == 1)[0]
        
        # Enforce maximum selections constraint
        if len(selected_sensors) > self.max_selections:
            # Keep only the first max_selections sensors
            selected_sensors = selected_sensors[:self.max_selections]
            action = np.zeros(self.num_sensors, dtype=int)
            action[selected_sensors] = 1
        
        # Calculate energy consumption
        energy_used = np.sum(action * self.sensor_energy_cost)
        
        # Update energy
        self.energy_remaining = max(0, self.energy_remaining - energy_used)
        
        # Calculate turbulence detection performance
        detection_accuracy, false_alarm_rate = self._calculate_detection_performance(action)
        
        # Calculate reward
        reward = self._calculate_reward(
            action, detection_accuracy, false_alarm_rate, energy_used
        )
        
        # Update environment state
        self._update_environment_state()
        
        self.current_step += 1
        self.total_reward += reward
        
        # Check termination conditions
        terminated = (
            self.current_step >= self.max_episode_steps or 
            self.energy_remaining <= 0
        )
        
        truncated = False  # We don't use truncation in this environment
        
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'detection_accuracy': detection_accuracy,
            'false_alarm_rate': false_alarm_rate,
            'energy_used': energy_used,
            'sensors_selected': len(selected_sensors),
            'turbulence_present': self.turbulence_present
        })
        
        return obs, reward, terminated, truncated, info
    
    def _sample_flight_phase(self):
        """Sample current flight phase"""
        # Weight different phases by typical flight time distribution
        phase_weights = [0.05, 0.05, 0.15, 0.5, 0.15, 0.05, 0.05]  # cruise is most common
        return np.random.choice(self.flight_phases, p=phase_weights)
    
    def _generate_flight_state(self):
        """Generate realistic flight state based on current phase"""
        base_state = np.random.normal(0, 1, self.flight_state_dim)
        
        # Modify based on flight phase
        if self.flight_phase == 'cruise':
            base_state[0] = np.random.normal(35000, 2000)  # altitude
            base_state[1] = np.random.normal(450, 50)      # airspeed
            base_state[2] = np.random.normal(0, 0.1)       # vertical_rate
        elif self.flight_phase == 'climb':
            base_state[0] = np.random.normal(20000, 5000)  # altitude
            base_state[1] = np.random.normal(300, 30)      # airspeed
            base_state[2] = np.random.normal(1500, 200)    # vertical_rate
        elif self.flight_phase == 'descent':
            base_state[0] = np.random.normal(15000, 5000)  # altitude
            base_state[1] = np.random.normal(280, 30)      # airspeed
            base_state[2] = np.random.normal(-800, 200)    # vertical_rate
        
        return base_state.astype(np.float32)
    
    def _generate_turbulence_scenario(self):
        """Generate turbulence scenario based on flight phase and weather"""
        # Turbulence probability varies by flight phase
        turbulence_probs = {
            'taxi': 0.01, 'takeoff': 0.05, 'climb': 0.15, 
            'cruise': 0.08, 'descent': 0.12, 'approach': 0.18, 'landing': 0.03
        }
        
        prob = turbulence_probs.get(self.flight_phase, 0.05)
        return np.random.random() < prob
    
    def _calculate_detection_performance(self, action):
        """Calculate turbulence detection accuracy and false alarm rate"""
        selected_sensors = np.where(action == 1)[0]
        
        if len(selected_sensors) == 0:
            return 0.0, 0.5  # No sensors selected = no detection capability
        
        # Base performance depends on selected sensors' informativeness and reliability
        sensor_quality = np.mean(self.sensor_informativeness[selected_sensors] * 
                                self.sensor_reliability[selected_sensors])
        
        # Adjust for flight phase (some sensors work better in certain phases)
        phase_multiplier = self._get_phase_sensor_effectiveness()
        effective_quality = sensor_quality * phase_multiplier
        
        # Calculate detection accuracy (higher when turbulence is present)
        if self.turbulence_present:
            detection_accuracy = min(0.95, 0.4 + 0.6 * effective_quality)
        else:
            detection_accuracy = max(0.05, 0.1 + 0.3 * effective_quality)
        
        # False alarm rate (should be low for good sensors)
        false_alarm_rate = max(0.01, 0.2 - 0.15 * effective_quality)
        
        return detection_accuracy, false_alarm_rate
    
    def _get_phase_sensor_effectiveness(self):
        """Get sensor effectiveness multiplier based on flight phase"""
        phase_effectiveness = {
            'taxi': 0.6, 'takeoff': 0.8, 'climb': 0.9,
            'cruise': 1.0, 'descent': 0.9, 'approach': 0.7, 'landing': 0.6
        }
        return phase_effectiveness.get(self.flight_phase, 1.0)
    
    def _calculate_reward(self, action, detection_accuracy, false_alarm_rate, energy_used):
        """Calculate reward based on detection performance and resource efficiency"""
        
        # Detection performance reward
        if self.turbulence_present:
            # Reward accurate detection of turbulence
            detection_reward = detection_accuracy * 20.0
            # Penalize false negatives heavily (safety critical)
            miss_penalty = (1 - detection_accuracy) * -30.0
        else:
            # Reward low false alarm rate when no turbulence
            detection_reward = (1 - false_alarm_rate) * 10.0
            # Penalize false alarms
            miss_penalty = false_alarm_rate * -15.0
        
        # Energy efficiency reward
        energy_penalty = energy_used * 0.1
        
        # Selection efficiency (penalize selecting too many or too few sensors)
        num_selected = np.sum(action)
        if num_selected == 0:
            selection_penalty = -20.0  # No sensors is very bad
        elif num_selected > self.max_selections:
            selection_penalty = -5.0 * (num_selected - self.max_selections)
        else:
            # Optimal range is 3-8 sensors
            optimal_range = range(3, 9)
            if num_selected in optimal_range:
                selection_penalty = 2.0
            else:
                selection_penalty = -1.0 * abs(num_selected - 5.5)
        
        # Combine all reward components
        total_reward = (detection_reward + miss_penalty - energy_penalty + 
                       selection_penalty)
        
        return total_reward
    
    def _update_environment_state(self):
        """Update environment state for next step"""
        # Update flight state (simulate flight progression)
        self.flight_state += np.random.normal(0, 0.1, self.flight_state_dim)
        
        # Potentially change flight phase
        if np.random.random() < 0.05:  # 5% chance to change phase
            current_idx = self.flight_phases.index(self.flight_phase)
            # Bias towards sequential phase transitions
            if current_idx < len(self.flight_phases) - 1:
                self.flight_phase = self.flight_phases[current_idx + 1]
        
        # Update turbulence scenario
        self.turbulence_present = self._generate_turbulence_scenario()
        
        # Simulate sensor degradation
        self.sensor_degradation += np.random.uniform(0, 0.001, self.num_sensors)
        self.sensor_reliability = np.maximum(0.5, 
            self.sensor_reliability - self.sensor_degradation)
        
        # Random sensor failures (rare)
        failure_prob = 0.001
        failures = np.random.random(self.num_sensors) < failure_prob
        self.sensor_availability[failures] = 0.0
        
        # Sensor recovery (also rare)
        recovery_prob = 0.01
        recoveries = np.random.random(self.num_sensors) < recovery_prob
        self.sensor_availability = np.where(recoveries, 1.0, self.sensor_availability)
    
    def _get_observation(self):
        """Get current observation vector"""
        time_normalized = self.current_step / self.max_episode_steps
        energy_normalized = self.energy_remaining / self.energy_budget
        energy_levels = np.full(self.num_sensors, energy_normalized)
        
        obs = np.concatenate([
            self.flight_state,
            self.sensor_availability,
            energy_levels,
            [time_normalized]
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self):
        """Get environment info dictionary"""
        return {
            'flight_phase': self.flight_phase,
            'energy_remaining': self.energy_remaining,
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'sensor_availability': self.sensor_availability.copy()
        }
    
    def render(self, mode='human'):
        """Render environment state (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Phase: {self.flight_phase}")
            print(f"Energy: {self.energy_remaining:.1f}, Turbulence: {self.turbulence_present}")
            print(f"Sensors Available: {np.sum(self.sensor_availability)}/{self.num_sensors}")
            print("-" * 50)


class SensorSelectionAgent(nn.Module):
    """
    Neural network agent for sensor selection using PPO algorithm
    """
    def _init_(self, obs_dim, num_sensors, hidden_dim=256):
        super()._init_()
        
        self.num_sensors = num_sensors
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head for sensor selection probabilities
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sensors),
            nn.Sigmoid()  # Output probabilities for each sensor
        )
        
        # Critic head for value estimation
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        """Forward pass through the network"""
        features = self.feature_extractor(obs)
        
        # Actor output: probabilities for each sensor
        sensor_probs = self.actor_head(features)
        
        # Critic output: state value
        value = self.critic_head(features)
        
        return sensor_probs, value
    
    def get_action(self, obs, deterministic=False):
        """Get action from current observation"""
        with torch.no_grad():
            sensor_probs, value = self(obs)
            
            if deterministic:
                # Select sensors with highest probabilities
                action = (sensor_probs > 0.5).float()
            else:
                # Stochastic selection based on probabilities
                from torch.distributions import Bernoulli
                dist = Bernoulli(sensor_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
                
                return action, log_prob, value
            
            return action, None, value