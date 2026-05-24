# Current SSL Crash-NPC Training Issue

## Goal

Train a controlled NPC to create realistic `side-swipe-left` crashes against a replay ego vehicle in GPUDrive.

The desired behavior is:

- Ego follows its original Waymo logged trajectory.
- NPC drives naturally in the same road/lane context.
- NPC approaches from ego's left side.
- Contact happens after a visible approach, not immediately at episode start.
- Videos should be long enough to show ego motion and NPC approach.

## What Was Going Wrong Initially

The original policy learned an exploit:

- NPC spun or pivoted unnaturally.
- It only optimized the final collision geometry.
- Episodes often ended almost immediately.
- The ego did not have enough time to visibly move.

The classifier only evaluates geometry at collision time, so PPO could satisfy the terminal reward without producing a realistic trajectory.

## Changes Already Made

### Pair-only filtered scenes

`scene_filter.py` now selects a viable NPC/ego pair and writes pair-only scenes:

- Object `0`: controlled NPC.
- Object `1`: replay ego.
- Other vehicles are removed.

This removed the previous `non_target:*` collision problem, where the NPC hit unrelated traffic.

### Metadata cleanup

Pair-only scenes now reset metadata:

- `sdc_track_index = 0`
- `tracks_to_predict = []`
- `objects_of_interest = []`

This fixed GPUDrive warnings about invalid `track_index` values after removing extra objects.

### Longer-start filtering

SSL/SSR scene filtering now requires:

- larger longitudinal offset, currently `|lx| in [14m, 30m]`
- moving vehicles, currently at least `2.0 m/s`
- long valid replay horizon, currently at least `80` valid steps

This prevents scenes where the NPC and ego start almost touching.

### Minimum-duration reward gate

`ppo_env.py` currently uses:

- `MIN_SUCCESS_STEP = 45`
- `MIN_EARLY_SUCCESS_REWARD = 3.0`

Early correct SSL crashes get a smaller graded reward, while full-duration SSL crashes get the full terminal reward.

### Dense SSL/SSR shaping

Additional shaping was added for side-swipe behavior:

- reward being on the correct ego side
- reward lateral offset near side-swipe distance
- reward heading alignment
- reward closing longitudinal gap
- penalize rear-end geometry

## Current Observed Problem

After making episodes longer, the task became too hard when success was purely terminal.

Recent logs showed:

```text
crash_rate ~= 0
mean_len ~= 86-88
labels mostly rear-end / side-swipe-right
too_early:side-swipe-left only rising slowly
```

This means:

- Episodes are now long enough.
- Ego motion is visible in principle.
- But PPO is not reliably discovering the correct SSL behavior.
- The target signal is still too sparse or too weak relative to wrong crash modes.

## Current Hypothesis

The dataset/scene setup is now closer to correct, but the reward is still the bottleneck.

The policy needs stronger pre-contact guidance to learn:

1. stay on ego's left side,
2. preserve same-direction heading,
3. close the longitudinal gap gradually,
4. avoid rear-end alignment,
5. make side contact after a visible approach.

The latest dense shaping change is intended to provide this gradient.

## What To Check Next

Run a fresh SSL training job from scratch using the current code:

```bash
.venv/bin/python -m collision_classifier.ppo_train \
  --crash_type ssl \
  --data_dir data/processed/filtered_pair_long/training_ssl \
  --dataset_size 278 \
  --num_worlds 64 \
  --total_steps 500000 \
  --resample_scenes
```

Stop early if by `120k` steps:

- `too_early:side-swipe-left` is not clearly rising, or
- full `side-swipe-left` is not starting to appear, or
- `rear-end` / `side-swipe-right` still dominate.

If the labels improve, render videos and inspect:

- video duration,
- ego motion,
- whether NPC stays road-aligned,
- whether contact is actually on ego's left side,
- whether the approach looks like a realistic sideswipe rather than a pivot or bump.

## Important Convention

In the renderer:

- blue vehicle = controlled NPC
- gray/light vehicle = replay ego

For SSL:

- label means ego's left side is contacted
- NPC may contact using its right side

So "blue NPC right side hits ego" can still be a correct `side-swipe-left`.

