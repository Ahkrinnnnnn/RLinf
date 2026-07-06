# RC09 插管任务 — Pi0.5 RL 后训练

桌面插管任务：将 pink rod 插入 blue tube。  
Prompt：`Insert the pink rod into the blue tube`  
Gym ID：`Isaac-RC09-PegInsert-Visuomotor-v0`

SFT 在 LeRobot 完成；本目录仅包含 RLinf 侧的 RL 后训练配置与代码引用。

---

## 整体流程

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

可选分支：使用 `td3_pi05_rlt.yaml` 进行 RLT Stage-2（TD3+BC，论文 §IV-B）；`ppo_pi05_rlt.yaml` 已弃用并重定向到该配置。

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
export ISAACLAB_PATH=/path/to/IsaacLab
export RC09_URDF_DIR=/home/dingyuxuan/zyy/RC09-05_urdf.SLDASM
bash rlinf/envs/isaaclab/isaaclab_tasks/rc09_peg_insert/scripts/install_rc09_task.sh
```

---

## 环境安装

```bash
bash requirements/install.sh embodied --model openpi --env isaaclab
source .venv/bin/activate
# 然后运行 install_rc09_task.sh（见上）
```

依赖：Isaac Sim 5.1.0、LeRobot 训好的 Pi0.5 SFT checkpoint。

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
| 物体初始位置与随机 reset | `EventCfg.reset_peg` |
| 相机位姿 | `scene.table_cam` / `scene.wrist_cam` |
| 成功判定阈值 | `mdp/rewards.py` → `peg_insert_success` |
| 语言指令 | `env.yaml` → `init_params.task_description`（须与 LeRobot SFT 一致） |

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
| `use_rl_token: false` | 不加载 RLT（`ppo_pi05_rlt.yaml` 中为 true） |
| `noise_level` | flow-SDE 探索噪声强度 |

优化器：`actor.optim` 中的 `lr`（policy）、`value_lr`（critic）。

---

## 观测 / 动作契约

| 字段 | 维度 | 来源 |
|------|------|------|
| `main_images` | 256×256×3 | table_cam |
| `wrist_images` | 256×256×3 | wrist_cam |
| `states` | 7 | eef_pos(3) + axis_angle(3) + gripper(1) |
| `task_descriptions` | str | prompt |
| `actions` | 7 | xyz + rpy + gripper |

须与 LeRobot SFT 数据的 obs/action 格式一致。

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
