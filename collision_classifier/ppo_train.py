"""
PPO training for adversarial crash NPC using GPUDrive's native IPPO infrastructure.

Usage:
  cd /path/to/gpudrive
  .venv/bin/python -m collision_classifier.ppo_train --crash_type ssl
  .venv/bin/python -m collision_classifier.ppo_train --crash_type ssr
  .venv/bin/python -m collision_classifier.ppo_train --crash_type re

Outputs:
  collision_classifier/checkpoints/<crash_type>_ppo_best   — best crash-rate checkpoint
  collision_classifier/checkpoints/<crash_type>_ppo_last   — periodic checkpoint
"""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import torch
from stable_baselines3.common.callbacks import BaseCallback

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from gpudrive.integrations.sb3.ppo import IPPO
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy

from collision_classifier.ppo_env import CrashVecEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


class CrashCallback(BaseCallback):
    def __init__(
        self,
        env: CrashVecEnv,
        crash_type: str,
        log_every: int = 5_000,
        save_every: int = 50_000,
        resample_scenes: bool = False,
    ):
        super().__init__(verbose=0)
        self.env = env
        self.crash_type = crash_type
        self.log_every = log_every
        self.save_every = save_every
        self.resample_scenes = resample_scenes
        self.best_crash_rate = 0.0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        n = self.num_timesteps

        if n % self.log_every == 0 and n > 0:
            rate = self.env.crash_rate()
            contact_rate = self.env.target_contact_rate()
            elapsed = time.time() - self.t0
            sps = n / elapsed

            label_counts: dict = {}
            for l in self.env.crash_labels_buf[-200:]:
                label_counts[l] = label_counts.get(l, 0) + 1

            ep = len(self.env.crash_hits)
            mean_len = self.env.mean_episode_len()
            lane_align = self.env.mean_lane_align()
            print(
                f"step={n:>7}  ep={ep:>5}  crash_rate={rate:.3f}  "
                f"target_contact={contact_rate:.3f}  "
                f"mean_len={mean_len:.1f}  lane_align={lane_align:.3f}  "
                f"sps={sps:.0f}  labels={label_counts}"
            )

            if rate > self.best_crash_rate and ep >= 50:
                self.best_crash_rate = rate
                path = os.path.join(CHECKPOINT_DIR, f"{self.crash_type}_ppo_best")
                self.model.save(path)
                print(f"Saved → {path}.zip")

        if n % self.save_every == 0 and n > 0:
            path = os.path.join(CHECKPOINT_DIR, f"{self.crash_type}_ppo_last")
            self.model.save(path)

        return True

    def _on_rollout_end(self) -> None:
        if self.resample_scenes:
            self.env.swap_scenes()


def train(
    crash_type: str,
    num_worlds: int = 64,
    total_steps: int = 500_000,
    data_dir: str = CrashVecEnv.DATA_DIR,
    dataset_size: int = 50,
    seed: int = 42,
    log_every: int = 5_000,
    save_every: int = 50_000,
    device: str = _DEFAULT_DEVICE,
    resample_scenes: bool = False,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n=== PPO Training crash NPC: {crash_type.upper()} ===")
    print(f"  total_steps={total_steps}  num_worlds={num_worlds}  device={device}  resample_scenes={resample_scenes}\n")

    env = CrashVecEnv(
        crash_type=crash_type,
        num_worlds=num_worlds,
        data_dir=data_dir,
        dataset_size=dataset_size,
        seed=seed,
        device=device,
    )

    # n_steps = one full episode per world per rollout
    n_steps = env.episode_len  # 91
    batch_size = max(1, (num_worlds * n_steps) // 5)  # 5 minibatches

    # Minimal exp_config stub — IPPO stores it but only checks resample_scenes
    exp_config = SimpleNamespace(resample_scenes=False)

    model = IPPO(
        n_steps=n_steps,
        batch_size=batch_size,
        env=env,
        seed=seed,
        verbose=0,
        device=device,
        mlp_class=FFN,
        policy=FeedForwardPolicy,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.01,       # entropy bonus — keeps policy exploring
        n_epochs=5,
        max_grad_norm=0.5,
        env_config=None,     # only needed for LateFusionNet
        exp_config=exp_config,
    )

    callback = CrashCallback(
        env=env,
        crash_type=crash_type,
        log_every=log_every,
        save_every=save_every,
        resample_scenes=resample_scenes,
    )

    model.learn(total_timesteps=total_steps, callback=callback)

    path = os.path.join(CHECKPOINT_DIR, f"{crash_type}_ppo_last")
    model.save(path)
    print(f"\nDone. Best crash rate: {callback.best_crash_rate:.3f}")
    print(f"Final checkpoint: {path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash_type",    type=str, required=True, choices=["ssl", "ssr", "re"])
    parser.add_argument("--num_worlds",    type=int, default=64)
    parser.add_argument("--total_steps",   type=int, default=500_000)
    parser.add_argument("--data_dir",      type=str, default=CrashVecEnv.DATA_DIR)
    parser.add_argument("--dataset_size",  type=int, default=50)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--log_every",     type=int, default=5_000)
    parser.add_argument("--save_every",       type=int,  default=50_000)
    parser.add_argument("--device",           type=str,  default=_DEFAULT_DEVICE)
    parser.add_argument("--resample_scenes",  action="store_true",
                        help="Swap in new Waymo scenes after every PPO rollout for generalization")
    args = parser.parse_args()

    train(
        crash_type=args.crash_type,
        num_worlds=args.num_worlds,
        total_steps=args.total_steps,
        data_dir=args.data_dir,
        dataset_size=args.dataset_size,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        device=args.device,
        resample_scenes=args.resample_scenes,
    )
