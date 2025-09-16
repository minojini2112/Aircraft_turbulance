#!/usr/bin/env python3
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-cfg",   required=True, help="pretrain YAML")
    p.add_argument("--encoder-out",    default="models/pretrained/encoder.pth", help="Encoder path")
    p.add_argument("--rl-cfg",         required=True, help="RL YAML")
    p.add_argument("--agent-out",      default="models/rl/ppo_agent.zip", help="RL agent path")
    p.add_argument("--tensorboard",    default="logs/rl_tb", help="RL TensorBoard dir")
    args = p.parse_args()

    # Step 1: Self-supervised pretraining
    subprocess.run([
        "python3", "scripts/training/run_pretraining.py",
        "--config", args.pretrain_cfg,
        "--output", args.encoder_out
    ], check=True)

    # Step 2: RL sensor selection training
    subprocess.run([
        "python3", "scripts/training/run_rl_training.py",
        "--config", args.rl_cfg,
        "--output", args.agent_out,
        "--tensorboard", args.tensorboard
    ], check=True)

    print("✅ Full training pipeline finished.")

if _name_ == "_main_":
    main()