# Docker Setup

本文档说明如何使用 Docker 构建、运行 openpi 环境，以及如何在容器内验证环境是否正确。

## 前置条件

| 依赖 | 安装参考 |
|---|---|
| Docker Engine (rootless) | [安装指南](https://docs.docker.com/engine/install/) / [rootless 模式](https://docs.docker.com/engine/security/rootless/) |
| NVIDIA Container Toolkit | [安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| NVIDIA GPU Driver | 宿主机需安装 NVIDIA 驱动 |

> **注意**：
> - `snap` 安装的 Docker 与 NVIDIA Container Toolkit 不兼容（[issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)），请用 `sudo snap remove docker` 卸载。
> - Docker Desktop 同样与 NVIDIA runtime 不兼容（[issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)），请用 `sudo apt remove docker-desktop` 卸载。

---

## 文件结构

```
docker/
├── Dockerfile                        # 镜像定义
├── entrypoint.sh                     # 入口脚本（支持 --verify 标志）
├── docker-compose.headless.yaml      # 无 GUI 模式
├── docker-compose.x11.yaml           # X11 GUI 模式
└── README.md                         # 本文档的符号链接（可选）
```

---

## 快速开始

### 1. 构建 & 启动容器

**Headless 模式**（训练 / 推理 / 策略服务）：

```bash
docker compose -f docker/docker-compose.headless.yaml up --build -d
```

**X11 GUI 模式**（需要渲染窗口，如 LIBERO、MuJoCo 可视化）：

```bash
# 先在宿主机授权 X11
xhost +local:docker

docker compose -f docker/docker-compose.x11.yaml up --build -d
```

### 2. 进入容器

```bash
# Headless
docker exec -it openpi-dev-headless bash

# X11
docker exec -it openpi-dev-gui bash
```

### 3. 停止 & 清理

```bash
docker compose -f docker/docker-compose.headless.yaml down
# 或
docker compose -f docker/docker-compose.x11.yaml down
```

---

## 验证环境

提供了多种方式来验证容器内的环境是否正确安装。

### 方式 A：使用 `--verify` 标志（推荐）

entrypoint 内置了 `--verify` 参数，会自动运行 `tests/test_env_verification.py`：

```bash
# 在已运行的容器中执行
docker exec openpi-dev-headless /usr/local/bin/entrypoint.sh --verify

# 或者用 docker compose run（一次性启动 + 验证 + 退出）
docker compose -f docker/docker-compose.headless.yaml run --rm openpi-dev --verify
```

### 方式 B：使用 `verify-env` profile（一键验证）

```bash
docker compose -f docker/docker-compose.headless.yaml --profile test run --rm verify-env
```

这会启动一个临时容器，运行所有环境验证测试，结束后自动删除。

### 方式 C：手动在容器内运行

```bash
docker exec -it openpi-dev-headless bash

# 在容器 shell 中：
python -m pytest tests/test_env_verification.py -v
```

### 验证项目

`tests/test_env_verification.py` 会检查以下内容：

| 测试项 | 说明 |
|---|---|
| `test_python_version` | Python >= 3.11 |
| `test_openpi_importable` | openpi 包可导入 |
| `test_openpi_training_config` | 训练配置（`get_config`）可用 |
| `test_openpi_policy_config` | 策略配置可用 |
| `test_openpi_shared_download` | 下载工具可用 |
| `test_jax_importable` | JAX 可运行计算 |
| `test_jax_device` | JAX 检测到设备（CPU/GPU） |
| `test_pytorch_importable` | PyTorch 可创建张量 |
| `test_pytorch_cuda_optional` | 报告 CUDA 可用性 |
| `test_key_dependencies` | flax, numpy, transformers 可导入 |
| `test_openpi_transforms` | openpi.transforms 可用 |

期望输出示例：

```
tests/test_env_verification.py::test_python_version PASSED
tests/test_env_verification.py::test_openpi_importable PASSED
tests/test_env_verification.py::test_jax_importable PASSED
tests/test_env_verification.py::test_jax_device PASSED
tests/test_env_verification.py::test_pytorch_importable PASSED
tests/test_env_verification.py::test_pytorch_cuda_optional PASSED
tests/test_env_verification.py::test_key_dependencies PASSED
...
```

---

## 运行推理测试（需要网络下载 checkpoint）

```bash
docker exec -it openpi-dev-headless bash

# 完整推理测试（会下载 checkpoint，需要网络和 GPU）
python -m pytest tests/test_inference.py -v -m manual
```

---

## 在容器中使用 Policy Server

```bash
# 终端 1：启动策略服务
docker exec -it openpi-dev-headless bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/openpi_assets/checkpoints/pi05_droid

# 终端 2：测试客户端（host 网络模式，端口直通）
docker exec -it openpi-dev-headless bash
python examples/simple_client/main.py
```

> 由于使用了 `network_mode: host`，容器内的端口与宿主机共享，可直接从宿主机或其他容器访问。

---

## 常用环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENPI_DATA_HOME` | `~/.cache/openpi` | checkpoint 缓存目录（映射到容器内 `/openpi_assets`） |
| `DISPLAY` | 宿主机 `$DISPLAY` | X11 GUI 模式需要 |
| `IS_DOCKER` | `true` | 标识当前在容器中运行 |
| `JAX_PLATFORMS` | (auto) | 无 GPU 时设为 `cpu` |

---

## 故障排查

| 问题 | 解决方案 |
|---|---|
| `--verify` 后 JAX GPU 测试失败 | 检查 NVIDIA driver 和 nvidia-container-toolkit 是否正确安装 |
| `uv sync` 构建失败 | 删除缓存 `docker builder prune`，重新构建 |
| X11 模式无法显示窗口 | 确认已运行 `xhost +local:docker`，检查 `$DISPLAY` 环境变量 |
| 容器内 import 报错 | 进入容器运行 `uv sync && uv pip install -e .` 重新同步 |
| Checkpoint 下载失败 | 检查网络连接；确认 `OPENPI_DATA_HOME` 卷挂载正确 |
