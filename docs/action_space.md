# π 模型 Action 输出形式说明

本文档说明 openpi 中不同 π 模型的 action 输出形式。所有模型均为 **Joint Space（关节空间）** 输出，**没有** Cartesian / End-Effector 空间形式。

---

## 按平台/Checkpoint 分类

### 1. DROID 系列（π0-DROID、π0-FAST-DROID、π0.5-DROID）

| 项目 | 说明 |
|------|------|
| **Action 形式** | **Joint velocity（关节速度）** + Gripper position |
| **维度** | 8：7 关节速度 + 1 夹爪位置 |
| **控制频率** | 15 Hz |

```python
# examples/droid/main.py
env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
# action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
```

- 关节：速度 (rad/s)
- 夹爪：位置 [0=open, 1=closed]

---

### 2. ALOHA 系列（π0-ALOHA-towel、tupperware、pen-uncap）

| 项目 | 说明 |
|------|------|
| **Action 形式** | **Absolute joint position（绝对关节位置）** |
| **维度** | 14：左臂 6 关节 + 夹爪 1 + 右臂 6 关节 + 夹爪 1 |
| **控制频率** | 50 Hz |

```text
Action space: [left_arm_qpos (6),   # absolute joint position
               left_gripper_positions (1),
               right_arm_qpos (6),
               right_gripper_positions (1)]
```

- 关节：弧度 (radians)
- 夹爪：[0, 1]，0=open，1=closed

---

### 3. LIBERO 系列（π0.5-LIBERO）

| 项目 | 说明 |
|------|------|
| **Action 形式** | **Joint position delta（关节位置增量）** |
| **维度** | 7：6 关节 delta + 1 夹爪（或 7 关节 delta） |

训练配置注释："In Libero, the raw actions in the dataset are already delta actions"。模型预测的是相对当前 joint state 的**增量**，环境直接执行。

---

### 4. PolaRiS DROID jointpos 变体

部分 PolaRiS 配置（如 `pi05_droid_jointpos_polaris`）使用 **joint position（绝对关节位置）** 而非 velocity，因为训练数据使用 `DroidActionSpace.JOINT_POSITION`。这是 DROID 平台上的特例。

---

## 通用 Action Space 定义（norm_stats.md）

```
"dim_0:dim_5":   left arm joint angles
"dim_6":         left arm gripper position
"dim_7:dim_12":  right arm joint angles（双手时）
"dim_13":        right arm gripper position

7-DoF 机器人（如 Franka）: dim_0:6 为关节，dim_7 为夹爪
```

- 关节角度单位：弧度 (radians)
- 夹爪：[0.0, 1.0]，0=fully open，1=fully closed

---

## 快速对照表

| 平台 | Action 空间 | 关节含义 | 夹爪含义 |
|------|-------------|----------|----------|
| DROID（标准） | Joint velocity + gripper position | 7 维关节速度 | 1 维 [0,1] |
| ALOHA | Absolute joint position | 12 维关节角度 + 2 夹爪 | [0,1] |
| LIBERO | Joint position delta | 6/7 维关节增量 | 通常包含在 7 维内 |
| PolaRiS DROID jointpos | Absolute joint position | 7 维关节角度 | 1 维 [0,1] |

---

## 结论

**所有 openpi π 模型的策略输出均为 Joint Space**，无 Cartesian / End-Effector 形式。
