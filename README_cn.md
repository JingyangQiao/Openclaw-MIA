# Openclaw-MIA：Agent 工具调用的在线强化学习系统

基于环境反馈的二值过程奖励信号，进行 Agent 工具调用场景的在线强化学习训练。

## 系统架构

系统跨三个角色进行分布式协作：

| 角色 | 核心文件 | 职责 |
|------|----------|------|
| **训练机** | `run_remote_rl.sh`, `remote_rollout.py`, `convert_and_update.sh` | 运行 GRPO 策略梯度训练（Ray + SLIME），转换检查点，热更新 SGLang |
| **服务器A**（数据桥接） | `server_a.py`, `sglang_manager.sh`, `sglang_hot_reload.sh` | 收集 MIA Planner / OpenClaw 对话数据，接收训练好的权重，管理本地 SGLang 服务 |
| **服务器B**（数据缓冲） | `server_b.py`, `push_weight.py` | 缓冲收集到的数据供训练机拉取，将训练好的权重推送回服务器A |

### 数据流向

```
[MIA Planner / OpenClaw]
        │  对话数据
        ▼
[服务器A: server_a.py :6333]
        │  POST /receive
        ▼
[服务器B: server_b.py :3]
        │  GET /get_data
        ▼
[训练机: remote_rollout.py → SLIME train_async.py]
        │  保存检查点
        ▼
[convert_and_update.sh → HF 格式]
        │  SGLang 热更新 / push_weight 推送到服务器A
        ▼
[SGLang 推理服务更新]
```

![openclaw-rl-mia](.\openclaw-rl-mia.png)

## 方法概述

策略模型部署为 OpenAI 兼容的聊天代理。外部环境通过该代理发送多轮对话。对于每个主线轮次，系统执行以下流程：

1. 将请求转发给策略模型（由 SGLang 提供服务），收集响应及逐 token 的 log-probabilities。
2. 当下一轮到达时，其 user/环境消息作为上一轮的"下一状态"。
3. **过程奖励模型（PRM）** 根据下一状态评判上一轮响应质量，通过多数投票（`m` 次独立评估）对每轮打分：`+1`（好）、`-1`（差）或 `0`（中性）。
4. 多数投票得分作为该轮的标量奖励。
5. 没有下一状态的轮次不参与训练（`loss_mask = 0`），除非它是会话中唯一的轮次。

### 优势估计（GRPO）

使用 **Group Relative Policy Optimization** 计算优势。对于标量奖励 `r` 的样本，优势均匀广播到所有响应 token：

$$A_t = r, \quad \forall t \in \text{response tokens}$$

不进行奖励归一化。

### 策略梯度损失

标准 PPO 风格的非对称裁剪目标（$\varepsilon = 0.2$, $\varepsilon_{\text{high}} = 0.28$）：

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\Big[\min\!\big(\rho_t A_t,\ \text{clip}(\rho_t,\, 1{-}\varepsilon,\, 1{+}\varepsilon_{\text{high}}) \cdot A_t\big)\Big]$$

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

其中 $\beta_{\text{KL}} = 0.02$，熵奖励禁用。

## 文件结构

```
mia-code/
├── README.md                          # 英文文档
├── README_cn.md                       # 本文件（中文文档）
│
│  ── 训练机 ──
├── run_remote_rl.sh                   # 主启动脚本（Ray + SLIME 训练作业）
├── remote_rollout.py                  # 从服务器B拉取训练数据，转换为 SLIME Sample 格式
├── convert_and_update.sh              # DCP 检查点转 HF 格式 + 触发 SGLang 热更新
│
│  ── 服务器A（数据桥接，端口 6333）──
├── server_a.py                        # Flask API：收集对话数据 → 转发至服务器B，接收权重
├── collect_service.py                 # 后台守护进程：基于文件的对话收集
├── conversation_monitor_local.py      # 配对 user/assistant 消息并保存为本地 jsonl 文件
├── sglang_manager.sh                  # SGLang 服务管理（启动/停止/热更新）
├── sglang_hot_reload.sh              # 监控 HF 权重目录，检测到新权重后触发 SGLang 热更新
│
│  ── 服务器B（数据缓冲，端口 3）──
├── server_b.py                        # Flask API：数据缓冲区，供训练机拉取
├── push_weight.py                     # 将训练好的权重推送回服务器A
│
│  ── 工具 ──
└── openclaw_forward.py                # jsonl 文件 → conversation_complete 格式转换 + 转发至服务器B
```

## 如何运行

### 一些必要仓库

此项目需要 https://github.com/Gen-Verse/OpenClaw-RL/tree/main/slime 和 https://github.com/Gen-Verse/OpenClaw-RL/tree/main/Megatron-LM。
您需要下载这两个文件，并将它们与整个项目一同放入同一个文件夹中。

### 训练机

```bash
cd slime
bash ../mia-code/run_remote_rl.sh
```

关键环境变量：`NUM_GPUS`、`ACTOR_GPUS`、`ROLLOUT_GPUS`、`PRM_GPUS`、`SERVER_A_URL`、`SERVER_B_URL`、`HF_CKPT`、`SAVE_CKPT`。

### 服务器A

```bash
python server_a.py
# 运行在端口 6333
```

### 服务器B

```bash
python server_b.py
# 运行在端口 3
```

### 数据转发（jsonl → 服务器B）

```bash
# 监控目录并转发 jsonl 文件
python openclaw_forward.py /path/to/jsonl_dir --server-b-url http://server-b:3

# 生成模拟数据用于测试
python openclaw_forward.py mock 5
python openclaw_forward.py mock continuous 10
```

## 依赖

- **训练机**：`slime`、`Megatron-LM`、`transformers`、`ray`、`sglang`
- **服务器A/B**：`flask`、`requests`、`waitress`（可选，用于生产环境）


## 致谢

OpenClaw-RL 项目 [https://github.com/Gen-Verse/OpenClaw-RL]