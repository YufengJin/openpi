# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

**This fork** adds installation optimizations and [policy-websocket](https://github.com/YufengJin/policy_websocket) integration for remote inference. Compatible with [role-ros2](https://github.com/YufengJin/role-ros2) — robot learning full stack on ROS2. See [role-ros2 README](https://github.com/YufengJin/role-ros2/blob/main/README.md) for policy deployment.

Models: π₀ (flow-based VLA), π₀-FAST (autoregressive), π₀.₅ (upgraded with knowledge insulation). Checkpoints: [gs://openpi-assets/checkpoints](https://console.cloud.google.com/storage/browser/openpi-assets/checkpoints).

## Requirements

| Mode       | Memory   | Example GPU |
| ---------- | -------- | ----------- |
| Inference  | > 8 GB   | RTX 4090    |
| Fine-Tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-Tuning (Full) | > 70 GB  | A100 80GB   |

Ubuntu 22.04 recommended.

## Installation

```bash
git clone --recurse-submodules https://github.com/YufengJin/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Docker

```bash
# Build and start (headless)
docker compose -f docker/docker-compose.headless.yaml up --build -d

# Verify environment
docker compose -f docker/docker-compose.headless.yaml run --rm openpi-dev --verify

# Enter container
docker exec -it openpi-headless bash
```

**Policy server in Docker**:

```bash
docker exec -it openpi-headless bash
uv run scripts/policy_server.py --port 8000
```

See [docs/docker.md](docs/docker.md) for X11 GUI mode, troubleshooting.

## Policy Server (policy-websocket)

Loads openpi DROID model and serves via WebSocket. Compatible with:

- [role-ros2](https://github.com/YufengJin/role-ros2) — robot learning full stack on ROS2
- [RoboCasa](https://robocasa.github.io/) — large-scale simulation benchmark
- [LIBERO](https://github.com/YufengJin/LIBERO) — lifelong robot learning benchmark

```bash
# Default: pi05_droid on port 8000
uv run scripts/policy_server.py --port 8000

# Custom config and checkpoint
uv run scripts/policy_server.py --config pi0_fast_droid --checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid
```

**Client usage** (use `--arm_controller joint_vel` and `--policy_server_addr localhost:8000`):

```bash
# RoboCasa
python robocasa/scripts/run_demo.py --arm_controller joint_vel --policy_server_addr localhost:8000

# LIBERO
python LIBERO/scripts/run_demo.py --arm_controller joint_vel --policy_server_addr localhost:8000
```

Output: action_dim 8 (joint_vel 7 + gripper).

**Dependency**: `uv add "policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git"`

## Inference (Quick)

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)
action_chunk = policy.infer(example)["actions"]
```

See [examples/inference.ipynb](examples/inference.ipynb), [examples/droid/README.md](examples/droid/README.md).

## More

- [Fine-tuning](examples/droid/README_train.md)
- [Remote inference](docs/remote_inference.md)
- [Upstream openpi](https://github.com/Physical-Intelligence/openpi) — full docs, PyTorch, troubleshooting
