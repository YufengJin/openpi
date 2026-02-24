#!/usr/bin/env python3
"""
Policy server for RoboCasa — loads an openpi DROID model and serves it via WebSocket.

Client sends raw robosuite obs; server remaps to DROID internally.
Output: action_dim 8 (joint_vel 7 + gripper). Use --arm_controller joint_vel on client.

Usage:
    python scripts/policy_server.py --port 8000
    python scripts/policy_server.py --config pi0_fast_droid --checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid

Client (use --arm_controller joint_vel):
    python LIBERO/scripts/run_demo.py --arm_controller joint_vel --policy_server_addr localhost:8000
    python robocasa/scripts/run_demo.py --arm_controller joint_vel --policy_server_addr localhost:8000
"""

import argparse
import contextlib
import logging
import socket
import sys

try:
    from policy_websocket import ActionChunkBroker, BasePolicy, WebsocketPolicyServer
except ImportError as e:
    raise ImportError("policy_websocket not found. Install with: uv add policy-websocket") from e

import numpy as np

logger = logging.getLogger(__name__)

BANNER_WIDTH = 60
DROID_IMG_SIZE = 224
DROID_TARGET_HW = (DROID_IMG_SIZE, DROID_IMG_SIZE)

# Keys that indicate obs has images (for init vs inference)
_IMAGE_KEYS = ("observation/exterior_image_1_left", "agentview_image", "robot0_agentview_left_image")

# Required keys for raw obs validation
RAW_LIBERO_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos")
RAW_ROBOCASA_KEYS = ("robot0_agentview_left_image", "robot0_eye_in_hand_image", "robot0_joint_pos", "robot0_gripper_qpos")


def _validate_keys(obs: dict, required: tuple, format_name: str) -> None:
    """Raise ValueError if any required key is missing."""
    missing = [k for k in required if k not in obs or obs[k] is None]
    if missing:
        raise ValueError(
            f"Observation format '{format_name}' requires keys {list(required)}. "
            f"Missing: {missing}. Received keys: {list(obs.keys())[:20]}..."
        )


def _ensure_uint8_hwc(img: np.ndarray, target_hw: tuple = DROID_TARGET_HW) -> np.ndarray:
    """Convert image to uint8 HWC and resize to target if needed."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    h, w = target_hw
    if img.shape[0] != h or img.shape[1] != w:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((w, h), resample=Image.BICUBIC))
    return img


def _gripper_qpos_to_droid(gripper_qpos: np.ndarray) -> np.ndarray:
    """Map Panda robot0_gripper_qpos (2,) to DROID gripper_position (1,) in [0,1]. 0=open, 1=closed."""
    q = np.asarray(gripper_qpos).flatten()
    val = (q[0] + q[1]) / 2.0 if len(q) >= 2 else float(q[0])
    return np.array([np.clip(val / 0.04, 0.0, 1.0)], dtype=np.float64)


def _prepare_obs_raw(
    exterior_img: np.ndarray,
    wrist_img: np.ndarray,
    joint_pos: np.ndarray,
    gripper_qpos: np.ndarray,
    prompt: str,
) -> dict:
    """Build DROID obs from images and proprio. joint_pos must be 7D."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64).flatten()
    if joint_pos.shape[0] != 7:
        raise ValueError(f"robot0_joint_pos must be 7D, got shape {joint_pos.shape}")
    exterior = _ensure_uint8_hwc(np.flipud(np.asarray(exterior_img)))
    wrist = _ensure_uint8_hwc(np.flipud(np.asarray(wrist_img)))
    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": _gripper_qpos_to_droid(gripper_qpos),
        "prompt": prompt,
    }


def prepare_obs_from_libero(raw_obs: dict) -> dict:
    """Convert raw LIBERO obs to DROID format."""
    _validate_keys(raw_obs, RAW_LIBERO_KEYS, "raw_libero")
    prompt = raw_obs.get("task_description", raw_obs.get("prompt", ""))
    return _prepare_obs_raw(
        raw_obs["agentview_image"],
        raw_obs["robot0_eye_in_hand_image"],
        raw_obs["robot0_joint_pos"],
        raw_obs["robot0_gripper_qpos"],
        prompt,
    )


def prepare_obs_from_robocasa(raw_obs: dict) -> dict:
    """Convert raw RoboCasa obs to DROID format."""
    _validate_keys(raw_obs, RAW_ROBOCASA_KEYS, "raw_robocasa")
    prompt = raw_obs.get("task_description", raw_obs.get("prompt", ""))
    return _prepare_obs_raw(
        raw_obs["robot0_agentview_left_image"],
        raw_obs["robot0_eye_in_hand_image"],
        raw_obs["robot0_joint_pos"],
        raw_obs["robot0_gripper_qpos"],
        prompt,
    )


def remap_obs_to_droid(obs: dict) -> dict:
    """
    Detect obs format and remap to DROID format.
    Supports: raw LIBERO, raw RoboCasa, DROID (already prepared).
    Raises ValueError if required keys are missing.
    """
    # Init: action_dim only, no images
    has_images = any(k in obs and obs.get(k) is not None for k in _IMAGE_KEYS)
    if "action_dim" in obs and not has_images:
        return obs

    # Already DROID
    if "observation/exterior_image_1_left" in obs and obs["observation/exterior_image_1_left"] is not None:
        return obs
    # Raw LIBERO
    if "agentview_image" in obs and obs["agentview_image"] is not None:
        return prepare_obs_from_libero(obs)
    # Raw RoboCasa
    if "robot0_agentview_left_image" in obs and obs["robot0_agentview_left_image"] is not None:
        return prepare_obs_from_robocasa(obs)
    raise ValueError(
        "Observation format not recognized. Expected one of: "
        "raw LIBERO (agentview_image, robot0_eye_in_hand_image, robot0_joint_pos, robot0_gripper_qpos), "
        "raw RoboCasa (robot0_agentview_left_image, robot0_eye_in_hand_image, robot0_joint_pos, robot0_gripper_qpos), "
        f"DROID (observation/exterior_image_1_left). Received keys: {list(obs.keys())[:25]}..."
    )


DEFAULT_CONFIGS = {
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
    "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
}


class OpenPIPolicyAdapter(BasePolicy):
    """Adapter: openpi policy returns infer(obs) -> dict with 'actions' (H, 8); BasePolicy compatible."""

    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs):
        return self._policy.infer(obs)

    def reset(self):
        if hasattr(self._policy, "reset"):
            self._policy.reset()


class ObsRemapWrapper(BasePolicy):
    """Remap raw LIBERO/RoboCasa obs to DROID format before passing to inner policy."""

    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs):
        droid_obs = remap_obs_to_droid(obs)
        return self._policy.infer(droid_obs)

    def reset(self):
        self._policy.reset()


class ResetOnInitPolicy(BasePolicy):
    """When client sends init (action_dim only, no images), reset and return zeros."""

    def __init__(self, policy):
        self._policy = policy

    def infer(self, obs):
        has_images = any(
            k in obs and obs.get(k) is not None
            for k in (*_IMAGE_KEYS, "primary_image")
        )
        if "action_dim" in obs and not has_images:
            self._policy.reset()
            return {"actions": np.zeros(int(obs["action_dim"]), dtype=np.float64)}
        return self._policy.infer(obs)

    def reset(self):
        self._policy.reset()


def _print_banner(args, config, checkpoint_dir, policy_metadata, local_ip):
    """Print server startup banner."""
    action_horizon = config.model.action_horizon
    sep = "=" * BANNER_WIDTH
    print(f"\n{sep}")
    print("OpenPI Policy Server for RoboCasa (policy-websocket)")
    print(sep)
    print(f"  config:     {args.config}")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  model:      {config.model.model_type}")
    print(f"  action_dim: {config.model.action_dim} (raw) → 8 (DROID output)")
    print(f"  horizon:    {action_horizon}")
    print(f"  host:       {args.host}:{args.port}")
    print(f"  ip:         {local_ip}")
    print(f"  metadata:   {policy_metadata}")
    print(sep)
    print(f"Waiting for connections on ws://{args.host}:{args.port} ...")
    print("Press Ctrl+C to stop.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenPI policy server for RoboCasa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pi05_droid",
        help=f"OpenPI config name. Pre-configured: {list(DEFAULT_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (local or gs://). If None, uses the default for --config.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt when client doesn't send one.",
    )
    parser.add_argument("--record", action="store_true", help="Record policy I/O for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()

    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint_dir = args.checkpoint or DEFAULT_CONFIGS.get(args.config)
    if checkpoint_dir is None:
        print(f"ERROR: No default checkpoint for config '{args.config}'. Use --checkpoint.", file=sys.stderr)
        sys.exit(1)

    config = _config.get_config(args.config)

    logger.info("Loading model: config=%s, checkpoint=%s", args.config, checkpoint_dir)
    print("Loading model... This may take a few minutes on first run (downloading checkpoint).")

    inner_policy = _policy_config.create_trained_policy(config, checkpoint_dir, default_prompt=args.default_prompt)
    policy_metadata = inner_policy.metadata
    if args.record:
        inner_policy = _policy.PolicyRecorder(inner_policy, "policy_records")

    action_horizon = config.model.action_horizon
    adapter = OpenPIPolicyAdapter(inner_policy)
    broker = ActionChunkBroker(adapter, action_horizon=action_horizon)
    policy = ResetOnInitPolicy(ObsRemapWrapper(broker))

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Host: %s, IP: %s", hostname, local_ip)

    metadata = {
        **policy_metadata,
        "server_type": "openpi_robocasa",
        "config": args.config,
        "action_space": "joint_velocity",
        "action_dim": 8,
        "action_dims": 8,
        "action_horizon": action_horizon,
    }

    _print_banner(args, config, checkpoint_dir, policy_metadata, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )

    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    print("Server stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
