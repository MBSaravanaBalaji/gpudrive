"""
Interactive keyboard-controlled driving in GPUDrive.

Ego (agent 0) = keyboard controlled.
All other agents in the scene = log-replay (follow recorded Waymo trajectories).

Controls:
  W / UP    — accelerate forward
  S / DOWN  — brake / reverse
  A / LEFT  — steer left  (while coasting)
  D / RIGHT — steer right (while coasting)
  Combine W/S + A/D for steering while accelerating
  SPACE     — coast (zero accel, zero steer)
  R         — reset episode
  Q / ESC   — quit

The sim is rendered as a bird's-eye view via matplotlib (Agg) → numpy → OpenCV window.
"""

import torch
import numpy as np
import cv2

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = "data/processed/examples"
EPISODE_LEN = 91
ZOOM_RADIUS = 60      # metres around ego to show
WINDOW_NAME = "GPUDrive — WASD to drive | R=reset | Q=quit"

# Classic dynamics: accel ∈ linspace(−4,4,7), steer ∈ linspace(−π,π,13)
# action_index = accel_idx * 13 + steer_idx
N_STEER = 13
A_BRAKE, A_COAST, A_FWD   = 0, 3, 6          # accel indices
S_LEFT, S_SL, S_STR, S_SR, S_RIGHT = 0, 3, 6, 9, 12   # steer indices

def act(accel_idx: int, steer_idx: int) -> int:
    return accel_idx * N_STEER + steer_idx

COAST = act(A_COAST, S_STR)

# ── Environment ───────────────────────────────────────────────────────────────

def make_env() -> GPUDriveTorchEnv:
    config = EnvConfig(
        dynamics_model="classic",
        collision_behavior="stop",    # keep ego in scene on collision
        max_controlled_agents=1,      # only agent 0 is RL-controlled; rest = log-replay
        num_worlds=1,
        ego_state=True,
        road_map_obs=False,
        partner_obs=True,
        norm_obs=False,
    )
    data_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=1,
        dataset_size=5,
        shuffle=True,
        file_prefix="tfrecord",
    )
    return GPUDriveTorchEnv(
        config=config,
        data_loader=data_loader,
        max_cont_agents=1,
        device="cpu",                 # macOS = CPU only
        action_type="discrete",
        render_config=RenderConfig(),
    )

# ── Rendering ─────────────────────────────────────────────────────────────────

def render_frame(env: GPUDriveTorchEnv, step: int, total: int) -> np.ndarray:
    """Render current sim state → BGR numpy image for OpenCV."""
    figs = env.vis.plot_simulator_state(
        env_indices=[0],
        center_agent_indices=[0],
        zoom_radius=ZOOM_RADIUS,
    )
    fig = figs[0]
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    import matplotlib.pyplot as plt
    plt.close(fig)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Overlay step counter
    label = f"Step {step}/{total}"
    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr

# ── Keyboard → action mapping ─────────────────────────────────────────────────

def keys_to_action(pressed: set) -> int:
    fwd   = ord("w") in pressed or 82 in pressed   # 82 = UP arrow in cv2
    back  = ord("s") in pressed or 84 in pressed   # 84 = DOWN
    left  = ord("a") in pressed or 81 in pressed   # 81 = LEFT
    right = ord("d") in pressed or 83 in pressed   # 83 = RIGHT

    if fwd and left:   return act(A_FWD,   S_SL)
    if fwd and right:  return act(A_FWD,   S_SR)
    if back and left:  return act(A_BRAKE, S_SL)
    if back and right: return act(A_BRAKE, S_SR)
    if fwd:            return act(A_FWD,   S_STR)
    if back:           return act(A_BRAKE, S_STR)
    if left:           return act(A_COAST, S_SL)
    if right:          return act(A_COAST, S_SR)
    return COAST

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("Building environment…")
    env = make_env()
    env.reset()

    n_agents = env.max_agent_count
    step = 0
    pressed = set()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 700, 700)

    print("Window open. Click it and use WASD to drive.")

    while True:
        # Render
        frame = render_frame(env, step, EPISODE_LEN)
        cv2.imshow(WINDOW_NAME, frame)

        # Poll key (non-blocking, ~33 ms so we get ~30 fps UI response)
        key = cv2.waitKey(33) & 0xFF

        if key == ord("q") or key == 27:   # Q or ESC → quit
            break
        elif key == ord("r"):
            env.reset()
            step = 0
            pressed.clear()
            continue
        elif key == 255:                   # no key pressed
            pass
        elif key in (0, 82, 83, 84, 81):  # modifier / arrow: track pressed state
            # cv2.waitKey gives the most recent key; hold tracking requires extra work.
            # We treat each waitKey result as "currently held" for one step.
            pressed = {key}
        else:
            pressed = {key}

        # Build action tensor: shape (num_worlds=1, max_agent_count)
        action_idx = keys_to_action(pressed)
        actions = torch.zeros(1, n_agents, dtype=torch.long)
        actions[0, 0] = action_idx

        env.step_dynamics(actions)
        step += 1

        # Clear pressed so untouched keys don't persist across frames
        # (cv2 only reports one key per waitKey call, not held state)
        if key != 255:
            pressed.clear()

        # Auto-reset on done or episode end
        done = env.get_dones()
        if done[0, 0].item() or step >= EPISODE_LEN:
            info = env.get_infos()
            collided = info.collided[0, 0].item() if hasattr(info, "collided") else "?"
            goal     = info.goal_achieved[0, 0].item() if hasattr(info, "goal_achieved") else "?"
            print(f"Episode done at step {step} — collided={collided}  goal={goal}. Resetting…")
            env.reset()
            step = 0

    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
