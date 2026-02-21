#!/usr/bin/env python3
"""
Policy server for RoboCasa — loads an openpi DROID model and serves it via WebSocket.

The server accepts observations in **openpi DROID format** and returns action chunks.
The RoboCasa client (scripts/run_eval.py, scripts/run_demo.py) handles the mapping
between RoboCasa env observations and this format.

OpenPI DROID observation format (what the client should send):
    observation/exterior_image_1_left : uint8 (224, 224, 3)
    observation/wrist_image_left      : uint8 (224, 224, 3)
    observation/joint_position        : float (7,)
    observation/gripper_position      : float (1,)
    prompt                            : str

OpenPI DROID action output:
    actions : float (action_horizon, 8)
        dims 0-6 : joint velocity  (rad/s)
        dim  7   : gripper position [0=open, 1=closed]

Usage (from openpi container):
    # Default: pi05_droid
    python scripts/policy_server.py

    # With a specific config and checkpoint:
    python scripts/policy_server.py \
        --config pi0_fast_droid \
        --checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid

    # With a local fine-tuned checkpoint:
    python scripts/policy_server.py \
        --config pi05_droid \
        --checkpoint /openpi_assets/checkpoints/pi05_droid

Then on the RoboCasa side:
    python scripts/run_eval.py --policy_server_addr <openpi-host>:8000 --task_name PnPCounterToCab

    # With DROID obs format and joint_vel control:
    python scripts/run_eval.py --droid --policy_server_addr <openpi-host>:8000 --task_name PnPCounterToCab
"""

import argparse
import logging
import socket
import sys

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = {
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
    "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
}


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
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    checkpoint_dir = args.checkpoint or DEFAULT_CONFIGS.get(args.config)
    if checkpoint_dir is None:
        print(f"ERROR: No default checkpoint for config '{args.config}'. Use --checkpoint.", file=sys.stderr)
        sys.exit(1)

    config = _config.get_config(args.config)

    logger.info("Loading model: config=%s, checkpoint=%s", args.config, checkpoint_dir)
    print(f"Loading model: config={args.config}, checkpoint={checkpoint_dir}")
    print("This may take a few minutes on first run (downloading checkpoint)...")

    policy = _policy_config.create_trained_policy(
        config, checkpoint_dir, default_prompt=args.default_prompt
    )
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Host: %s, IP: %s", hostname, local_ip)

    print(f"\n{'=' * 60}")
    print(f"OpenPI Policy Server for RoboCasa")
    print(f"{'=' * 60}")
    print(f"  config:     {args.config}")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  model:      {config.model.model_type}")
    print(f"  action_dim: {config.model.action_dim} (raw) → 8 (DROID output)")
    print(f"  horizon:    {config.model.action_horizon}")
    print(f"  host:       {args.host}:{args.port}")
    print(f"  ip:         {local_ip}")
    print(f"  metadata:   {policy_metadata}")
    print(f"{'=' * 60}")
    print(f"Waiting for connections on ws://{args.host}:{args.port} ...")
    print("Press Ctrl+C to stop.\n")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            **policy_metadata,
            "server_type": "openpi_robocasa",
            "config": args.config,
            "action_space": "joint_velocity",
            "action_dims": 8,
            "action_horizon": config.model.action_horizon,
        },
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
