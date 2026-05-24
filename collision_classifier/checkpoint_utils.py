"""Helpers for warm-starting IPPO checkpoints across observation sizes."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from stable_baselines3.common.save_util import load_from_zip_file

from gpudrive.integrations.sb3.ppo import IPPO
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy


def checkpoint_input_dim(checkpoint_path: str, device: str = "cpu") -> int:
    """Return feature dim (first Linear in) from a saved IPPO policy."""
    _, params, _ = load_from_zip_file(checkpoint_path, device=device)
    policy_state = params["policy"]
    for key, tensor in policy_state.items():
        if key.endswith("actor_net.0.weight"):
            return int(tensor.shape[1])
    raise ValueError(f"Could not infer obs dim from {checkpoint_path}")


def _expand_first_linear(old_w: torch.Tensor, new_w: torch.Tensor) -> torch.Tensor:
    """Copy overlapping input columns; leave new columns at init."""
    out = new_w.clone()
    cols = min(old_w.shape[1], new_w.shape[1])
    out[:, :cols] = old_w[:, :cols]
    return out


def load_ippo_with_obs_expand(
    checkpoint_path: str,
    env,
    *,
    device: str,
    n_steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
) -> IPPO:
    """Load IPPO weights, expanding the first layer when obs grew (e.g. spawn_cond)."""
    ckpt_dim = checkpoint_input_dim(checkpoint_path, device=device)
    env_dim = env.obs_dim

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
        learning_rate=learning_rate,
        ent_coef=0.03,
        n_epochs=5,
        max_grad_norm=0.5,
        env_config=None,
        exp_config=SimpleNamespace(resample_scenes=False),
    )

    data, params, _ = load_from_zip_file(checkpoint_path, device=device)
    old_state = params["policy"]
    new_state = model.policy.state_dict()

    for key, new_tensor in new_state.items():
        if key not in old_state:
            continue
        old_tensor = old_state[key]
        if old_tensor.shape == new_tensor.shape:
            new_state[key] = old_tensor
        elif (
            key.endswith(".weight")
            and old_tensor.ndim == 2
            and old_tensor.shape[0] == new_tensor.shape[0]
            and old_tensor.shape[1] == ckpt_dim
            and new_tensor.shape[1] == env_dim
            and env_dim > ckpt_dim
        ):
            new_state[key] = _expand_first_linear(old_tensor, new_tensor)

    model.policy.load_state_dict(new_state, strict=False)
    model.learning_rate = learning_rate
    if "num_timesteps" in data:
        model.num_timesteps = int(data["num_timesteps"])
    if params and "learning_rate" in params:
        model.learning_rate = learning_rate

    print(
        f"Expanded checkpoint obs {ckpt_dim}→{env_dim} from {checkpoint_path}  "
        f"start_steps={model.num_timesteps}"
    )
    return model
