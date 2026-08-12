# RLT Stage-2 双框架协同 — 后续改进计划

本文档记录 **路径 1（共享核心）** 已完成部分与 **路径 2（Ray worker 集成）** 的待办项。  
共享数学与网络已落地并通过 conformance 测试；本文档不重复 README 中的 API 说明。

合并 upstream 后，RLinf 官方 RLT（`rlinf/algorithms/rlt/`，`loss_type: rlt_ac`）与 LeRobot 桥接层（`rlinf/rlt/`、`RLTokenBridge`，`loss_type: embodied_rlt_td3`）并存，职责见 RC09 README「RLT 集成架构」一节。

---

## 已完成（Phase 1 — 共享核心）

| 项 | 位置 | 状态 |
|----|------|------|
| TD3+BC 损失自由函数 | `lerobot.rl.algorithms.rlt_td3.losses` | ✅ |
| 统一共享入口 | `lerobot.rlt.shared` | ✅ |
| lerobot 单元测试 | `lerobot/tests/rl/test_rlt_shared_core.py` | ✅ 26 passed |
| RLinf 薄驱动 | `rlinf.rlt.RLTTD3Driver` | ✅ |
| RLinf chunked rollout 骨架 | `rlinf.rlt.ChunkedRolloutRunner` | ✅ |
| 跨框架 conformance | `RLinf/tests/unit_tests/test_rlt_shared_core_conformance.py` | ✅ 10 passed |
| RC09 TD3 任务配置 | `examples/embodiment/config/rc09_peg_insert/td3_pi05_rlt.yaml` | ✅ 配置就绪 |
| 环境：py3.12 + fork 安装 | `requirements/install.sh`（`LEROBOT_PATH` / `install_lerobot_runtime_deps`） | ✅ |

---

## Phase 2 — RLinf Ray worker 集成（仿真端到端训练）

目标：让 `bash examples/embodiment/run_async.sh rc09_peg_insert/td3_pi05_rlt` 真正跑通 Isaac Lab 上的 RLT Stage-2。

1. **`AsyncEmbodiedRLTTD3Policy` worker**  
   - 在 `rlinf/workers/actor/` 新增 FSDP/async actor worker。  
   - 内部持有 `RLTTD3Driver`（或等价包装），`update_step` 消费 replay buffer 批次。  
   - Critic ensemble 仅 learner 侧持有（与 lerobot `RLTTD3Algorithm` 一致）。

2. **`train_async.py` 注册 `embodied_rlt_td3`**  
   - 替换当前 `NotImplementedError` 为真实 runner + worker 分支。  
   - 复用 `AsyncEmbodiedRunner` + `WeightSyncer`（与 SAC async 同模式）。

3. **Rollout worker：chunk 开环 + Machine A 查询**  
   - 在 `AsyncMultiStepRolloutWorker`（或 RC09 专用子类）中：  
     - chunk 边界调用冻结 OpenPI + `RLTokenBridge` 提取 `z_rl` / `ref_chunk`；  
     - `RLTTD3Driver.predict_chunk_mean` 或 rollout actor 推理；  
     - 开环执行 `chunk_len` 步；  
     - 用 `ChunkedRolloutRunner` / `build_replay_windows` 写入 replay buffer（`replay_stride`、`SOURCE_*` 标签）。

4. **Replay buffer 批次格式对齐**  
   - 确保 worker 侧 `complementary_info` 含 `ref_chunk`、`ref_chunk_next`、`source`、`bc_target`，与 `lerobot.rl.algorithms.rlt_td3.losses` 的 `fb` 契约一致。

5. **OpenPI ↔ ChunkActor 权重边界**  
   - 仅同步 `ChunkActor` 权重（`get_actor_weights` / `load_actor_weights`），不同步冻结 VLA。  
   - Checkpoint 目录结构与 lerobot `RLTActorPolicy` 兼容，便于 sim→real 加载。

6. **RC09 仿真验证**  
   - `install_rc09_task.sh` + `td3_pi05_rlt.yaml` 端到端 smoke（短步数）。  
   - 对比 lerobot `golden_loss_values()` 与仿真跑若干 learner step 的 loss 量级（非 bit 级，但应同阶）。

---

## Phase 3 — 真机 ↔ 仿真闭环

1. **权重互转脚本**  
   - `lerobot/rlt/export_actor_for_rlinf.py`（或双向）：safetensors ↔ `RLTTD3Driver.get_actor_weights()` 格式。

2. **lerobot Stage-2 加载 RLinf checkpoint**  
   - `RLTStage2Orchestrator` 支持从 RLinf 导出目录 `load_weights`。

3. **超参 / 配置双向校验**  
   - 单测：`td3_pi05_rlt.yaml` 字段 ↔ `RLTTrainingConfig` / `RLTTD3AlgorithmConfig` 默认值一致。

---

## Phase 4 — 工程与运维

1. **`install.sh` 一体化**  
   - `embodied --model openpi --env isaaclab` 自动检测 sibling `../lerobot` 并 `install_lerobot_runtime_deps`（已实现逻辑，需在干净环境复验）。

2. **删除 `.venv.old`**  
   - 旧 3.11 环境部分文件属 `root`，需本机执行：  
     `sudo rm -rf RLinf/.venv.old`

3. **CI**  
   - lerobot：`test_rlt_shared_core.py`  
   - RLinf：`test_rlt_shared_core_conformance.py`（无需 Isaac/GPU）

4. **PLD 异步对齐（可选）**  
   - `lerobot.rl.async_runner.AsyncActorLearnerRunner` 接入 `PLDStage1Orchestrator.rl_train`（SERL 式异步）。

---

## 配置入口

| 场景 | 配置 | 启动命令（Phase 2 完成后） |
|------|------|---------------------------|
| RC09 仿真 RLT Stage-2 | `rc09_peg_insert/td3_pi05_rlt.yaml` | `run_async.sh rc09_peg_insert/td3_pi05_rlt` |
| RC09 PPO（无 RLT） | `rc09_peg_insert/ppo_pi05.yaml` | `run_embodiment.sh rc09_peg_insert/ppo_pi05` |
| 真机 RLT Stage-2 | `lerobot/examples/rlt/rlt_pi05_stage2_*.json` | `lerobot-rlt-stage2` |

已弃用：`ppo_pi05_rlt.yaml` → 重定向至 `td3_pi05_rlt.yaml`。
