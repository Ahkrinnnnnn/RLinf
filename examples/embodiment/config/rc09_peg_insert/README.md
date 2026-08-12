# RC09 插管任务 — Pi0.5 RL 后训练

桌面插管任务：将 pink rod 插入 blue tube。  
Prompt：`Insert the pink rod into the blue tube`  
Gym ID：`Isaac-RC09-PegInsert-Visuomotor-v0`

SFT 在 LeRobot 完成；本目录仅包含 RLinf 侧的 RL 后训练配置与代码引用。

---

## 整体流程

本项目在 LeRobot 完成 Pi0.5 SFT 后，于 RLinf 侧进行 RC09 仿真 RL 后训练。可选路线如下：

| 路线 | 配置 | 启动方式 | 状态 |
|------|------|----------|------|
| PPO 基线 | `ppo_pi05.yaml` | `run_embodiment.sh rc09_peg_insert/ppo_pi05` | 可用 |
| **冻结策略 Eval** | `evaluations/isaaclab/rc09_peg_insert_pi05_eval.yaml` | `evaluations/run_eval.sh isaaclab rc09_peg_insert_pi05_eval` | 可用 |
| RLT Stage-2（TD3+BC） | `td3_pi05_rlt.yaml` | `run_async.sh rc09_peg_insert/td3_pi05_rlt` | Ray worker 未接线（见 `rlinf/rlt/ROADMAP.md`） |
| 真机 Stage-2 | lerobot `lerobot-rlt-stage2` | 不经过本目录 | 与 RC09 仿真独立 |

PPO 基线流程：

```
LeRobot SFT（Pi0.5 checkpoint）
        │
        ▼
① 安装 Isaac Lab 任务（install_rc09_task.sh）
        │
        ▼
② 编辑 ppo_pi05.yaml，填入 checkpoint 路径
        │
        ▼
③ run_embodiment.sh rc09_peg_insert/ppo_pi05
        │
        ▼
TensorBoard 监控 env/success_once
```

RLT Stage-2 使用 `td3_pi05_rlt.yaml`（论文 TD3+BC）；`ppo_pi05_rlt.yaml` 已弃用并重定向至该配置。Stage-1 RL token 由 lerobot `lerobot-train` 训练，经 `rlt_bridge.py` 加载。

---

## 目录结构

```
examples/embodiment/config/rc09_peg_insert/
├── README.md
├── env.yaml              # 环境：task_id、prompt、episode、相机
├── ppo_pi05.yaml         # Pi0.5 → PPO
├── td3_pi05_rlt.yaml     # Pi0.5 + RL Token → TD3+BC（RLT Stage-2，论文对齐）
└── ppo_pi05_rlt.yaml     # 已弃用，重定向至 td3_pi05_rlt.yaml
```

---

## 两层环境代码说明

RC09 任务涉及两个目录，职责不同：

| 目录 | 路径 | 运行环境 | 职责 |
|------|------|----------|------|
| **Isaac Lab 任务** | `rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/` | Isaac Sim 子进程 | 物理仿真：场景、物体、reward、相机、MDP |
| **RLinf wrapper** | `rlinf/envs/isaaclab/tasks/rc09_peg_insert.py` | RLinf Ray worker | 将 Isaac obs 转为 RLinf 统一格式，供 rollout 使用 |

数据流：

```
Isaac Sim (isaaclab_tasks/)
    │  gym.make("Isaac-RC09-PegInsert-Visuomotor-v0")
    ▼
SubProcIsaacLabEnv（子进程隔离）
    │  raw obs: eef_pos, table_cam, ...
    ▼
IsaaclabRC09PegInsertEnv._wrap_obs()（tasks/rc09_peg_insert.py）
    │  main_images, wrist_images, states, task_descriptions
    ▼
RolloutWorker → OpenPI Pi0.5 推理 → action
    ▼
EmbodiedRunner PPO 更新
```

`isaaclab_tasks/` 下的代码通过 `install_rc09_task.sh` **软链接到 Isaac Lab fork**；修改 RLinf 源码后无需重新 install。`tasks/` 下的 wrapper 留在 RLinf 包内，由训练框架直接 import。

---

## install_rc09_task.sh 作用

路径：`rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh`

该脚本完成三件事：

1. **软链接任务代码**：`ln -s` 将 RLinf 源码目录链到  
   `$ISAACLAB_PATH/source/isaaclab_tasks/.../rc09_peg_insert/`
2. **修补 URDF 路径**：把 ROS `package://` mesh 路径替换为本地绝对路径，生成 `.isaac.urdf`
3. **注册 Gym 任务**：在 Isaac Lab 的 `isaaclab_tasks/__init__.py` 追加 import

**何时需要运行：** 首次部署、更换 `ISAACLAB_PATH`、换机器。  
**何时不需要：** 日常修改 `isaaclab_tasks/` 内的场景、reward、MDP（软链接实时生效）。

未运行此脚本时，RLinf 训练会在 env worker 启动 Isaac 时报 task not found。

```bash
export ISAACLAB_PATH=../IsaacLab
export RC09_URDF_DIR=../RC09-05_urdf.SLDASM
bash rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh
```

---

## 环境安装

Isaac Sim 5.1 二进制版只提供 **cp311** 原生扩展，请用 **Python 3.11** 创建 RLinf venv（与官方 Isaac Lab 示例一致）：

```bash
cd RLinf
bash requirements/install.sh embodied --model openpi --env isaaclab --python 3.11.11
source .venv/bin/activate

export ISAAC_PATH=../isaac_sim
export ISAACLAB_PATH=../IsaacLab
export RC09_URDF_DIR=../RC09-05_urdf.SLDASM

# 每次新终端（与 RLinf Isaac Lab 文档一致）
source $ISAAC_PATH/setup_conda_env.sh

bash rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh
```

依赖：Isaac Sim 5.1.0、Python 3.11 venv、LeRobot 训好的 Pi0.5 SFT checkpoint。

### Isaac Sim 路径

路径均相对 **RLinf 仓库根目录**（或通过环境变量覆盖）：

| 变量 | 示例 |
|------|------|
| `ISAAC_PATH` | `../isaac_sim` |
| `ISAACLAB_PATH` | `../IsaacLab` |
| `RC09_URDF_DIR` | `../RC09-05_urdf.SLDASM` |

Isaac Lab 需能解析 `_isaac_sim`（通常为 `IsaacLab/_isaac_sim` → `isaac_sim` 的软链接）。

---

## 启动 RL 后训练

### Pi0.5 checkpoint 路径

`ppo_pi05.yaml` 中 `model_path` 使用相对路径（相对**启动训练时的 cwd**）

### Pi0.5（标准 PPO）

编辑 `ppo_pi05.yaml`：

```yaml
actor.model.model_path: /path/to/lerobot-pi05-sft
rollout.model.model_path: /path/to/lerobot-pi05-sft
```

```bash
export EMBODIED_PATH=$(pwd)/examples/embodiment
bash examples/embodiment/run_embodiment.sh rc09_peg_insert/ppo_pi05
```

### Pi0.5 + RL Token（RLT Stage-2，TD3+BC）

编辑 `td3_pi05_rlt.yaml` 中的 `model_path` 与 `rl_token_checkpoint`，然后：

```bash
export EMBODIED_PATH=$(pwd)/examples/embodiment
bash examples/embodiment/run_async.sh rc09_peg_insert/td3_pi05_rlt
```

> **当前状态**：共享核心与 conformance 测试已就绪；Ray worker（`embodied_rlt_td3`）集成见 `rlinf/rlt/ROADMAP.md` Phase 2。在 worker 接线完成前，上述命令会提示 `NotImplementedError`。

监控：

```bash
tensorboard --logdir ../results --port 6006
```

关键指标：`env/success_once`

### 训练日志路径

通过 `run_embodiment.sh` 启动时，日志写入：

```
RLinf/logs/<时间戳>-rc09_peg_insert/ppo_pi05/
├── run_embodiment.log
├── tensorboard/all/
├── checkpoints/global_step_<N>/actor/
└── video/eval/          # eval 开启 save_video 时
```

直接运行 `train_embodied_agent.py` 时使用 yaml 默认路径 `../results/`。

---

## 冻结策略 Eval（只 rollout，不训练）

与 LIBERO / Polaris 等任务相同，RC09 使用 RLinf 官方 **`EmbodiedEvalRunner`**：只启动 **Env + Rollout**，不加载 Actor、不做梯度更新。

### 机制（与库内其他仿真一致）

| 组件 | 训练 (`ppo_pi05.yaml`) | Eval (`rc09_peg_insert_pi05_eval.yaml`) |
|------|------------------------|----------------------------------------|
| Runner | `EmbodiedRunner` | `EmbodiedEvalRunner` |
| 入口脚本 | `train_embodied_agent.py` | `evaluations/eval_embodied_agent.py` |
| Worker | actor + env + rollout | **仅 env + rollout** |
| 模型来源 | `actor.model` | **`rollout.model`（冻结）** |
| `task_type` | `embodied` | **`embodied_eval`** |

Eval 时 `env.eval` 典型设置：

- `ignore_terminations: True` — 跑满 `max_episode_steps`，不因成功提前截断 rollout 统计
- `auto_reset: True` — episode 结束后自动 reset，继续采样
- `rollout_epoch` × `total_num_envs` — 总评测 episode 数（默认 5×32=160）

### 启动前准备

与训练相同：安装依赖、`install_rc09_task.sh`、LeRobot SFT checkpoint。

### 编辑 checkpoint 路径

文件：`evaluations/isaaclab/rc09_peg_insert_pi05_eval.yaml`

```yaml
rollout.model.model_path: /path/to/lerobot-pi05-sft
```

### 运行

```bash
cd RLinf
export EMBODIED_PATH=$(pwd)/examples/embodiment
export ISAACLAB_PATH=/path/to/IsaacLab
export RC09_URDF_DIR=/path/to/RC09-05_urdf.SLDASM

# 安装任务（首次）
bash rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh

# 冻结策略 eval
bash evaluations/run_eval.sh isaaclab rc09_peg_insert_pi05_eval \
  rollout.model.model_path=/path/to/lerobot-pi05-sft
```

也可直接调用 Python：

```bash
python evaluations/eval_embodied_agent.py \
  --config-path evaluations/isaaclab \
  --config-name rc09_peg_insert_pi05_eval \
  rollout.model.model_path=/path/to/lerobot-pi05-sft
```

### 输出指标

终端与 TensorBoard 会打印 `eval/*` 前缀的聚合指标（由 `compute_evaluate_metrics` 汇总）：

| 指标 | 含义 |
|------|------|
| `eval/success_once` | 任意一步获得稀疏成功奖励的比例（**主成功率**） |
| `eval/success_at_end` | episode 结束时是否触发 insert 成功终止（`ignore_terminations=True` 时仍记录） |
| `eval/episode_len` | 平均步数 |
| `eval/return` | 累计回报 |
| `eval/num_trajectories` | 评测 episode 总数 |

视频默认保存到 `${runner.logger.log_path}/video/eval/`。

### 常用 Hydra 覆盖

```bash
# 更多 episode：5 epoch × 64 env = 320 trajectories
bash evaluations/run_eval.sh isaaclab rc09_peg_insert_pi05_eval \
  env.eval.total_num_envs=64 \
  env.eval.rollout_epoch=5

# 关闭视频加速
bash evaluations/run_eval.sh isaaclab rc09_peg_insert_pi05_eval \
  env.eval.video_cfg.save_video=False
```

训练过程中周期性 eval：在 `ppo_pi05.yaml` 设 `runner.val_check_interval: 50`（会使用同一套 `env.eval` 配置，但仍会训练 actor）。

---

## 重要文件

| 层级 | 文件 | 用途 |
|------|------|------|
| 场景/仿真 | `isaaclab_tasks/rc09_peg_insert/rc09_peg_insert_env_cfg.py` | 桌面、粉杆、蓝管、相机、物体参数 |
| 奖励/终止 | `isaaclab_tasks/.../mdp/rewards.py` | 插入成功判定、shaping |
| 机器人 | `isaaclab_tasks/.../rc09_robot_cfg.py` | URDF 路径、关节限位 |
| 任务安装 | `isaaclab_tasks/.../scripts/install_rc09_task.sh` | 部署到 Isaac Lab fork |
| RLinf wrapper | `rlinf/envs/isaaclab/tasks/rc09_peg_insert.py` | obs 格式转换 |
| 任务注册 | `rlinf/envs/isaaclab/__init__.py` | task ID → wrapper 映射 |
| OpenPI I/O | `rlinf/models/embodiment/openpi/policies/isaaclab_rc09_policy.py` | obs/action ↔ Pi0.5 格式 |
| OpenPI dataconfig | `rlinf/models/embodiment/openpi/dataconfig/isaaclab_rc09_dataconfig.py` | config 名 `pi05_isaaclab_rc09_peg_insert` |
| RL Token 桥接 | `rlinf/models/embodiment/openpi/rlt_bridge.py` | Stage-1 RLT 权重 → `z_rl` |
| RLT TD3 驱动 | `rlinf/rlt/td3_driver.py` | 复用 lerobot 共享损失数学 |
| RLT 改进计划 | `rlinf/rlt/ROADMAP.md` | Phase 2 worker 集成待办 |
| RL 配置（TD3 RLT） | `config/rc09_peg_insert/td3_pi05_rlt.yaml` | TD3+BC 超参、replay、async |
| RL 配置（PPO） | `config/rc09_peg_insert/ppo_pi05.yaml` | PPO 超参、冻结、checkpoint |
| 环境配置 | `config/rc09_peg_insert/env.yaml` | prompt、episode 长度、相机分辨率 |

### 场景修改入口

| 修改内容 | 文件/位置 |
|----------|-----------|
| 粉杆/蓝管尺寸与颜色 | `rc09_peg_insert_env_cfg.py` → `RC09PegInsertSceneCfg` |
| 物体初始位置与随机 reset | `EventCfg.reset_peg` / `reset_tube` |
| Domain Randomization（VLA 后训练） | `EventCfg` + `enable_domain_randomization` |
| 相机内外参 | `rlinf/envs/isaaclab/camera_calibration.py` | 通用 hand-eye session 加载；RC09 绑定见 `rc09_peg_insert/camera_calibration.py` |
| 相机位姿 DR | `randomize_table_cam_extrinsics` | 仅 top 外参微扰 |
| 成功判定阈值 | `mdp/rewards.py` → `peg_insert_success` |
| 语言指令 | `env.yaml` → `init_params.task_description`（须与 LeRobot SFT 一致） |

### Domain Randomization（VLA 后训练）

训练默认开启（`env.yaml` → `init_params.enable_domain_randomization: true`）；Eval 在 `rc09_peg_insert_pi05_eval.yaml` 中设为 `false`。

| 类别 | Event 项 | 说明 |
|------|----------|------|
| 光照 | `randomize_light` | DomeLight 强度、色温、HDR 背景 |
| 桌面纹理 | `randomize_table_visual_material` | 木纹/石材/金属等 BaseColor |
| 物体颜色 | `randomize_peg_color` / `randomize_tube_color` | 保持粉/蓝主色域，与语言 prompt 一致 |
| 相机外参 | `randomize_table_cam_extrinsics` | 桌面相机位姿 ±3 mm / ±0.5° |
| 物理 | `peg_physics_material` / `peg_mass` | 摩擦系数、质量 ±20% |
| 物体位姿 | `reset_peg` / `reset_tube` | 每 episode 随机初始位姿 |

视觉类 DR 需要 `scene.replicate_physics=False`（已在 env cfg 中设置）。实现见 `mdp/events.py`。

---

## RL 策略与冻结配置

### PPO 算法超参

文件：`ppo_pi05.yaml` → `algorithm:`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `adv_type` | `gae` | 优势估计 |
| `loss_type` | `actor_critic` | PPO loss |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 折扣与 GAE |
| `clip_ratio_high/low` | 0.2 | PPO clip |
| `update_epoch` | 3 | 每轮 rollout 更新次数 |
| `reward_coef` | 1.0 | 环境 reward 缩放 |

### 环境奖励

| 文件 | 内容 |
|------|------|
| `mdp/rewards.py` | 奖励函数定义 |
| `RewardsCfg` | 各项 reward 权重 |
| `TerminationsCfg` | episode 终止条件 |
| `env.yaml` → `use_rel_reward` | 相对 reward 模式 |

### 模型侧（Pi0.5）

文件：`ppo_pi05.yaml` → `actor.model.openpi:`

| 字段 | 说明 |
|------|------|
| `train_expert_only: false` | 全量微调 VLA |
| `train_expert_only: true` | 冻结 VLM，只训 action expert + value head |
| `detach_critic_input: true` | critic 梯度与 actor 分离 |
| `use_rl_token: false` | 不加载 RLT（`ppo_pi05.yaml` 默认） |
| `use_rl_token: true` | 加载 lerobot Stage-1 checkpoint（`rlt_bridge.py`） |
| `use_rlt: true` | RLinf 内置 token 模块（权重与 lerobot checkpoint 不互通，见 Maniskill 官方配置） |
| `noise_level` | flow-SDE 探索噪声强度 |

优化器：`actor.optim` 中的 `lr`（policy）、`value_lr`（critic）。

---

## 观测 / 动作契约

与 LeRobot SFT（`insert_the_blue_tube` / `crp_arm`）对齐：

| 字段 | 维度 | 策略侧（OpenPI / LeRobot） | Isaac 仿真内部 |
|------|------|---------------------------|----------------|
| `states` | 6 | 关节角 **度**（j1…j6） | `arm_joint_pos` **弧度** |
| `actions` 前 6 维 | 6 | 绝对关节角 **度** | 绝对关节角 **弧度** |
| `actions` 第 7 维 | 1 | 夹爪 **GOT0** [0, 1000]（0=关，1000=开） | 二值 ±1（wrapper 阈值 500） |
| `main_images` | 640×480×3 | table_cam | table_cam |
| `wrist_images` | 640×480×3 | wrist_cam | wrist_cam |
| `task_descriptions` | str | prompt | prompt |

单位转换在 `rlinf/envs/isaaclab/rc09_real_robot_io.py`，由 `IsaaclabRC09PegInsertEnv` 在 `_wrap_obs` / `step` 中自动完成。

**安全 clip（手册限位，非 norm 范围）**：关节 clip 到 RC09-05 手册 deg 限位；夹爪 clip 到 GOT0 [0, 1000]。在 deg→rad 进仿真之前执行。

**Pi0.5 quantile norm**：仍用 SFT 数据集 q01/q99（`norm_stats.json`），与 clip 无关。

### RLT Stage-2 归一化（lerobot 真机 / 未来 RLinf 仿真）

| 量 | 归一化？ | 范围 / 空间 |
|----|----------|-------------|
| VLA 图像 + state + ref chunk | ✅ Pi0.5 preprocessor | 数据集 **q01/q99** → 约 [-1, 1] |
| `z_rl` | 在 norm 后的 prefix 上算 | 无额外 norm |
| **`proprio` (s_p)** | ❌ **原始度** | 真机 `j*.pos`，6D，**不做 quantile** |
| Actor 输出 / replay `action` | ✅ 与 VLA 同空间 | **normalized chunk** |
| 下发机器人 / 仿真 | postprocessor **unnormalize** 后 | deg + GOT0，再 clip 限位 |

RLT **不需要**给 `proprio` 单独做 quantile；也**不要**用手册限位替代 Pi0.5 的 norm_stats。仿真侧只需 wrapper 的 deg↔rad + **action/state clip 到手册**（已实现）。

---

## 测试

```bash
cd RLinf
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit_tests/test_rc09_peg_insert.py -v
```

---

## 故障排查

| 现象 | 处理 |
|------|------|
| Task not registered | 运行 `install_rc09_task.sh` |
| URDF mesh 找不到 | 检查 `RC09_URDF_DIR`，确认生成 `.isaac.urdf` |
| 机械臂不动 | `rc09_robot_cfg.py` 补关节限位 |
| RLT import 失败 | 安装 LeRobot：`pip install -e /path/to/lerobot` |
| OOM | 降低 `env.train.total_num_envs` 或启用 `actor.enable_offload` |

---

## RLT 集成架构（merge upstream 后）

fork 已合并 RLinf 官方 RLT（`rlinf/algorithms/rlt/`，`loss_type: rlt_ac`），与 LeRobot 桥接层并存：

| 组件 | 路径 | 用途 |
|------|------|------|
| 官方 RLT Stage-2 | `rlinf/algorithms/rlt/` + `fsdp_rlt_ac_policy_worker` | Maniskill / 真机；Ray worker 已接线 |
| 官方 Stage-1 token | `rlt_token_transformer.py`，配置项 `use_rlt` | RLinf SFT 内训练；权重格式独立于 lerobot |
| LeRobot Stage-1 桥接 | `rlt_bridge.py`，配置项 `use_rl_token` | 加载 lerobot-train 产出的 Stage-1 checkpoint |
| LeRobot Stage-2 驱动 | `rlinf/rlt/RLTTD3Driver` | 复用 `lerobot.rlt.shared`；RC09 配置为 `embodied_rlt_td3`（Phase 2 待接线） |

RC09 的 `td3_pi05_rlt.yaml` 采用 `use_rl_token` 路径，与 Maniskill 官方配置（`use_rlt`）相互独立，不可混用 checkpoint。
