#!/usr/bin/env python3
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.rl.sensor_env import SensorSelectionEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to RL config YAML")
    p.add_argument("--output", default="models/rl/ppo_agent.zip",
                   help="Where to save RL agent")
    p.add_argument("--tensorboard", default="logs/rl_tb",
                   help="TensorBoard log directory")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Create vectorized environments
    env_fns = [lambda: SensorSelectionEnv(
        num_sensors=cfg['env']['num_sensors'],
        max_selections=cfg['env']['max_selections'],
        flight_state_dim=cfg['env']['flight_state_dim'],
        max_episode_steps=cfg['env']['max_episode_steps'],
        energy_budget=cfg['env']['energy_budget']
    ) for _ in range(cfg['env']['n_envs'])]
    env = DummyVecEnv(env_fns)

    # Initialize PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg['rl']['lr'],
        n_steps=cfg['rl']['n_steps'],
        batch_size=cfg['rl']['batch_size'],
        n_epochs=cfg['rl']['n_epochs'],
        gamma=cfg['rl']['gamma'],
        verbose=1,
        tensorboard_log=args.tensorboard
    )

    # Train
    model.learn(total_timesteps=cfg['rl']['total_timesteps'])

    # Save
    model.save(args.output)
    print(f"RL agent saved to {args.output}")

if _name_ == "_main_":
    main()