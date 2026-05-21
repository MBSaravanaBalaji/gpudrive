"""
Training script for the adversarial crash NPC.

Usage:
  cd /path/to/gpudrive
  .venv/bin/python -m collision_classifier.train --crash_type ssl
  .venv/bin/python -m collision_classifier.train --crash_type ssr
  .venv/bin/python -m collision_classifier.train --crash_type re   (future)

Outputs:
  collision_classifier/checkpoints/<crash_type>_best.pt   — best crash-rate model
  collision_classifier/checkpoints/<crash_type>_last.pt   — latest checkpoint
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

from collision_classifier.dqn import DQNAgent, DQNConfig
from collision_classifier.env_wrapper import CrashEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train(
    crash_type: str,
    total_steps: int = 500_000,
    data_dir: str = CrashEnv.DATA_DIR,
    dataset_size: int = 50,
    seed: int = 42,
    log_every: int = 2_000,
    save_every: int = 50_000,
    device: str = "cpu",
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n=== Training crash NPC: {crash_type.upper()} ===")
    print(f"  total_steps={total_steps}  device={device}\n")

    env = CrashEnv(
        crash_type=crash_type,
        data_dir=data_dir,
        dataset_size=dataset_size,
        seed=seed,
        device=device,
    )

    cfg = DQNConfig(
        obs_dim=CrashEnv.OBS_DIM,
        n_actions=env.n_actions,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=50_000,
        batch_size=256,
        target_update_freq=500,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=int(total_steps * 0.8),  # keep exploration alive through most of training
        hidden=128,
        device=device,
    )
    agent = DQNAgent(cfg)

    # Rolling stats
    window = 100
    ep_rewards   = deque(maxlen=window)
    crash_hits   = deque(maxlen=window)   # 1 = correct crash, 0 = otherwise
    crash_labels = deque(maxlen=window)   # crash label strings

    best_crash_rate = 0.0
    ep = 0
    global_step = 0
    t0 = time.time()

    obs = env.reset()

    while global_step < total_steps:
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)

        agent.push(obs, action, reward, next_obs, done)
        loss = agent.train_step()

        obs = next_obs
        global_step += 1

        if done:
            ep += 1
            ep_rewards.append(reward)

            label = info.get("crash_label")
            if info.get("collided") and label is not None:
                crash_labels.append(label)
                hit = 1 if label in env.target_labels else 0
                crash_hits.append(hit)
            else:
                crash_hits.append(0)

            obs = env.reset()

        # ── Logging ──────────────────────────────────────────────────────────
        if global_step % log_every == 0 and ep > 0:
            crash_rate = sum(crash_hits) / max(len(crash_hits), 1)
            mean_reward = sum(ep_rewards) / max(len(ep_rewards), 1)
            elapsed = time.time() - t0
            sps = global_step / elapsed

            label_counts: dict = {}
            for l in crash_labels:
                label_counts[l] = label_counts.get(l, 0) + 1

            print(
                f"step={global_step:>7}  ep={ep:>5}  "
                f"crash_rate={crash_rate:.3f}  "
                f"mean_rew={mean_reward:+.3f}  "
                f"eps={agent.epsilon:.3f}  "
                f"sps={sps:.0f}  "
                f"labels={label_counts}"
            )

            # Save best
            if crash_rate > best_crash_rate and len(crash_hits) >= window // 2:
                best_crash_rate = crash_rate
                agent.save(os.path.join(CHECKPOINT_DIR, f"{crash_type}_best.pt"))

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if global_step % save_every == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"{crash_type}_last.pt"))

    # Final save
    agent.save(os.path.join(CHECKPOINT_DIR, f"{crash_type}_last.pt"))
    print(f"\nDone. Best crash rate: {best_crash_rate:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash_type",   type=str, required=True, choices=["ssl", "ssr", "re"])
    parser.add_argument("--total_steps",  type=int, default=500_000)
    parser.add_argument("--data_dir",     type=str, default=CrashEnv.DATA_DIR)
    parser.add_argument("--dataset_size", type=int, default=50)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--log_every",    type=int, default=2_000)
    parser.add_argument("--save_every",   type=int, default=50_000)
    parser.add_argument("--device",       type=str, default="cpu")
    args = parser.parse_args()

    train(
        crash_type=args.crash_type,
        total_steps=args.total_steps,
        data_dir=args.data_dir,
        dataset_size=args.dataset_size,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        device=args.device,
    )
