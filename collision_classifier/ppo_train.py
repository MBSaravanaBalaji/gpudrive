"""
PPO training for adversarial crash NPC using GPUDrive's native IPPO infrastructure.

See collision_classifier/TRAINING.md for full instructions (SSL / SSR / RE),
checkpoint naming, and how to avoid overwriting production models.

Usage:
  .venv/bin/python -m collision_classifier.ppo_train --crash_type ssl --checkpoint_tag multispawn ...
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
from gpudrive.networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy

from collision_classifier.ppo_env import CrashVecEnv
from collision_classifier.checkpoint_utils import (
    checkpoint_input_dim,
    load_ippo_with_obs_expand,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


class CrashCallback(BaseCallback):
    def __init__(
        self,
        env: CrashVecEnv,
        crash_type: str,
        log_every: int = 5_000,
        save_every: int = 50_000,
        resample_scenes: bool = False,
        finetune: bool = False,
        checkpoint_tag: str | None = None,
    ):
        super().__init__(verbose=0)
        self.env = env
        self.crash_type = crash_type
        self.log_every = log_every
        self.save_every = save_every
        self.resample_scenes = resample_scenes
        base = f"{crash_type}_ppo_{checkpoint_tag}" if checkpoint_tag else f"{crash_type}_ppo"
        suffix = "_finetune_best" if finetune else "_best"
        self.best_name = f"{base}{suffix}"
        if checkpoint_tag:
            self.last_name = f"{base}_last"
        elif finetune:
            self.last_name = f"{crash_type}_ppo_finetune_last"
        else:
            self.last_name = f"{crash_type}_ppo_last"
        self.best_crash_rate = 0.0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        n = self.num_timesteps

        if n % self.log_every == 0 and n > 0:
            rate = self.env.crash_rate()
            contact_rate = self.env.target_contact_rate()
            target_coll_rate = self.env.target_collision_rate()
            target_contact_coll_rate = self.env.target_contact_collision_rate()
            elapsed = time.time() - self.t0
            sps = n / elapsed

            label_counts: dict = {}
            for l in self.env.crash_labels_buf[-200:]:
                label_counts[l] = label_counts.get(l, 0) + 1

            ep = len(self.env.crash_hits)
            mean_len = self.env.mean_episode_len()
            lane_align = self.env.mean_lane_align()
            family_stats = self.env.spawn_family_stats()
            print(
                f"step={n:>7}  ep={ep:>5}  crash_rate={rate:.3f}  "
                f"target_collision={target_coll_rate:.3f}  "
                f"target_contact_collision={target_contact_coll_rate:.3f}  "
                f"target_contact={contact_rate:.3f}  "
                f"mean_len={mean_len:.1f}  lane_align={lane_align:.3f}  "
                f"sps={sps:.0f}  labels={label_counts}"
            )
            if family_stats:
                fam_parts = [
                    f"{fam}:{bucket['crash_rate']:.2f}({int(bucket['episodes'])})"
                    for fam, bucket in sorted(family_stats.items())
                ]
                print(f"         spawn_families  {'  '.join(fam_parts)}")

            if rate > self.best_crash_rate and ep >= 50:
                self.best_crash_rate = rate
                path = os.path.join(CHECKPOINT_DIR, self.best_name)
                self.model.save(path)
                print(f"Saved → {path}.zip")

        if n % self.save_every == 0 and n > 0:
            path = os.path.join(CHECKPOINT_DIR, self.last_name)
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
    load_checkpoint: str | None = None,
    learning_rate: float = 3e-4,
    spawn_cond: bool = False,
    spawn_family: str | None = None,
    checkpoint_tag: str | None = None,
    late_fusion: bool = False,
    reset_timesteps: bool = False,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n=== PPO Training crash NPC: {crash_type.upper()} ===")
    print(
        f"  total_steps={total_steps}  num_worlds={num_worlds}  device={device}  "
        f"resample_scenes={resample_scenes}  spawn_cond={spawn_cond}  "
        f"spawn_family={spawn_family}  checkpoint_tag={checkpoint_tag}  "
        f"load_checkpoint={load_checkpoint}  late_fusion={late_fusion}\n"
    )

    env = CrashVecEnv(
        crash_type=crash_type,
        num_worlds=num_worlds,
        data_dir=data_dir,
        dataset_size=dataset_size,
        seed=seed,
        device=device,
        spawn_cond=spawn_cond,
        spawn_family=spawn_family,
        late_fusion=late_fusion,
    )

    # n_steps = one full episode per world per rollout
    n_steps = env.episode_len
    batch_size = max(1, (num_worlds * n_steps) // 5)  # 5 minibatches

    exp_config = SimpleNamespace(
        resample_scenes=resample_scenes,
        ego_state_layers=[64, 32],
        road_object_layers=[64, 64],
        road_graph_layers=[64, 64],
        shared_layers=[64, 64],
        act_func="tanh",
        dropout=0.0,
        last_layer_dim_pi=64,
        last_layer_dim_vf=64,
        spawn_extra_dim=CrashVecEnv.OBS_SPAWN_DIM if spawn_cond and late_fusion else 0,
    )
    env_config = env.env_config if late_fusion else None
    mlp_class = LateFusionNet if late_fusion else FFN
    policy_class = LateFusionPolicy if late_fusion else FeedForwardPolicy

    if load_checkpoint:
        if late_fusion:
            model = IPPO.load(
                load_checkpoint,
                env=env,
                custom_objects={
                    "mlp_class": LateFusionNet,
                    "policy_class": LateFusionPolicy,
                },
                device=device,
            )
            model.learning_rate = learning_rate
            if model.n_steps != n_steps:
                print(f"  n_steps {model.n_steps}→{n_steps} (episode length changed)")
                model.n_steps = n_steps
                model.rollout_buffer.n_steps = n_steps
                model.rollout_buffer.reset()
        else:
            ckpt_dim = checkpoint_input_dim(load_checkpoint, device=device)
            if env.obs_dim > ckpt_dim:
                model = load_ippo_with_obs_expand(
                    load_checkpoint,
                    env,
                    device=device,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    seed=seed,
                    learning_rate=learning_rate,
                )
            else:
                model = IPPO.load(
                    load_checkpoint,
                    env=env,
                    custom_objects={"mlp_class": FFN, "policy_class": FeedForwardPolicy},
                    device=device,
                )
                model.learning_rate = learning_rate
                if model.n_steps != n_steps:
                    print(f"  n_steps {model.n_steps}→{n_steps} (episode length changed)")
                    model.n_steps = n_steps
                    model.rollout_buffer.n_steps = n_steps
                    model.rollout_buffer.reset()
        start_steps = model.num_timesteps
        if reset_timesteps:
            model.num_timesteps = 0
            start_steps = 0
        target_steps = start_steps + total_steps
        print(
            f"Warm start from {load_checkpoint}  lr={learning_rate}  "
            f"start_steps={start_steps}  target_steps={target_steps}"
            + ("  (timesteps reset)" if reset_timesteps else "")
        )
    else:
        start_steps = 0
        target_steps = total_steps
        model = IPPO(
            n_steps=n_steps,
            batch_size=batch_size,
            env=env,
            seed=seed,
            verbose=0,
            device=device,
            mlp_class=mlp_class,
            policy=policy_class,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            clip_range=0.2,
            learning_rate=learning_rate,
            ent_coef=0.03,
            n_epochs=5,
            max_grad_norm=0.5,
            env_config=env_config,
            exp_config=exp_config,
        )

    callback = CrashCallback(
        env=env,
        crash_type=crash_type,
        log_every=log_every,
        save_every=save_every,
        resample_scenes=resample_scenes,
        finetune=load_checkpoint is not None,
        checkpoint_tag=checkpoint_tag,
    )

    model.learn(
        total_timesteps=target_steps,
        callback=callback,
        reset_num_timesteps=reset_timesteps or load_checkpoint is None,
    )

    path = os.path.join(CHECKPOINT_DIR, callback.last_name)
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
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Warm-start from an existing IPPO checkpoint (.zip)")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--spawn_cond",
        action="store_true",
        help="Add spawn-family conditioning to obs (15-dim). Required for multi_spawn datasets.",
    )
    parser.add_argument(
        "--spawn_family",
        type=str,
        default=None,
        choices=["behind_near", "behind_far", "side_same", "side_opposite", "ahead"],
        help="Curriculum: train on one spawn family only (filters scene filenames).",
    )
    parser.add_argument(
        "--checkpoint_tag",
        type=str,
        default=None,
        help="Checkpoint name tag, e.g. multispawn → ssr_ppo_multispawn_best.zip",
    )
    parser.add_argument(
        "--late_fusion",
        action="store_true",
        help="Use GPUDrive road/partner obs with LateFusionNet (train from scratch)",
    )
    parser.add_argument(
        "--reset_timesteps",
        action="store_true",
        help="Load weights only: reset PPO step counter (fresh curriculum on warm start)",
    )
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
        load_checkpoint=args.load_checkpoint,
        learning_rate=args.learning_rate,
        spawn_cond=args.spawn_cond,
        spawn_family=args.spawn_family,
        checkpoint_tag=args.checkpoint_tag,
        late_fusion=args.late_fusion,
        reset_timesteps=args.reset_timesteps,
    )
