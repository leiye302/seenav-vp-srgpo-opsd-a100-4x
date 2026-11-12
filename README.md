# SeeNav-Agent / VP + SRGPO + OPSD / 4×A100 80GB

在作者公开的 **EmbodiedBench-Navigation / VP + SRGPO** 训练上，仅接入 Teacher-only OPSD。使用
[SeeNav-Agent 论文 v2](https://arxiv.org/pdf/2512.02631v2) 和
[作者代码 7ed6a59](https://github.com/WzcTHU/SeeNav-Agent/tree/7ed6a59052701951bdb2c7d51dd6d9103219cf3e)。
这是对照实验的复现仓库，不是作者发布，也不包含权重、数据副本或训练结果。

## 改动范围

保留[作者 SRGPO 启动脚本](https://github.com/WzcTHU/SeeNav-Agent/blob/7ed6a59052701951bdb2c7d51dd6d9103219cf3e/verl-agent/examples/srgpo_trainer/run_ebnav.sh)。
除四卡 `trainer.n_gpus_per_node: 8 → 4` 和隔离产物路径之外，只加入 OPSD 的评分及损失接线。
干净的官方 checkout 保持不变；`fetch` 把 [13 行补丁](patches/opsd.patch) 应用到另一个 checkout，
仅触及 `ray_trainer.py` 和 `dp_actor.py`。原模型、环境、数据、VP、动作、原生 SRGPO 优势、
原有 reference KL、优化器、batch、验证和保存设置均不变；没有切换 LoRA 或新增 offload。

| 项目 | 沿用作者公开的训练入口 |
| --- | --- |
| 模型与训练方式 | Qwen/Qwen2.5-VL-3B-Instruct；VP + SRGPO，FSDP，vLLM TP=2 |
| 一批的 rollout | 8 个任务，每任务 4 条轨迹（32 条），每条最多 20 步 |
| SRGPO | 跨 batch 内的步骤随机分组，每组 16；mean/std 归一化，步骤优势权重 0.5 |
| 优化 | LR 1e-6；全局 PPO mini batch 8，单卡 micro batch 1；KL 0.01 |
| 验证、保存 | 8 任务训练内验证，每 5 步；150 epochs、每 150 步保存 |
| 新增 OPSD | SDAR sampled-token loss，系数 0.01，gate beta 5，原响应 mask 上 token-mean |

论文原配是 8 张 H20 96GB。4 卡时，上游 `adjust_batch` 会按 4 卡而非 8 卡做原生整除补齐，
因此不承诺两个并行布局逐步数值相同。没有为此改写采样或 SRGPO 实现。

## OPSD 与原对照口径

1. Student 按原版 VP 输入采样并在 AI2-THOR 执行动作，原版 SRGPO 计算轨迹与步骤优势。
2. 更新前，同一 Actor 的无梯度视角给**同一批已生成 token**打分。Teacher 在原 prompt 前读到
   一段[静态导航技能](seenav_opsd/navigation_skill.txt)，保留原图像张量和所有原 token。
   在文本前缀后平移 RoPE 位置，不重新渲染、重采样或截断 Student 响应。技能只讲当前视图、
   朝向、动作反馈和绕障规则，不含 Oracle 动作、隐藏地图、未来观测或成功轨迹。
3. 采用[锁定的 SDAR 原生损失](https://github.com/ZJU-REAL/SDAR/blob/29dcb12a55d8fc17ba6ae03fd0a69048f43051f3/verl/trainer/ppo/sdar_utils.py)。
   在原响应掩码 $m_t$ 上，令 $d_t=\log p_{T,t}-\log p_{S,t}$、
   $g_t=\operatorname{sg}[\sigma(5d_t)]$。新增的项为：

   $$
   0.01\,\frac{\sum_t m_t g_t (\operatorname{sg}[\log p_{T,t}]-\log p_{S,t})}{\sum_t m_t}.
   $$

   这里 $\operatorname{sg}$ 表示停止梯度；原策略梯度、熵和参考 KL 保留。

Teacher 不参与 rollout 或验证，不是用于 reference KL 的冻结基模，不产生新动作。
没有 RETHINK、Analyzer、局部 hint 或额外模型 API。日志记录 `sdar/teacher_rows`、
`sdar/teacher_prefix_tokens`、gate 和 loss；Teacher 重评分需要额外一次 Actor 前向。

## 准备环境

在分配了 **4 张 A100 80GB** 的 Linux x86-64 服务器执行。准备 Python 3.10、Git、Bash、wget、
支持 CUDA 12.4 的 NVIDIA 驱动及 AI2-THOR CloudRendering 所需的 Vulkan/驱动库。容器须能访问
宿主机的 GPU 图形能力；先检查 `nvidia-smi` 与 `vulkaninfo --summary`。AI2-THOR 首次启动
会下载 Unity 资源，需允许出站访问并为本机缓存、HF 模型和完整 FSDP checkpoint 预留磁盘。

```bash
git clone https://github.com/leiye302/seenav-vp-srgpo-opsd-a100-4x.git
cd seenav-vp-srgpo-opsd-a100-4x
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools==75.1.0' wheel
python reproduce.py fetch
```

若服务器的代理不能传输 GitHub 仓库、而直连已获准，可以用 `python reproduce.py fetch --direct`。
也可以通过 `--source` 指定已经克隆到该服务器的干净官方 checkout，仍会校验同一个 commit。
不使用作者本机环境，也不需要任何仓库外的私有文件。

安装作者训练依赖。FSDP + vLLM 入口不需要安装未使用的 Megatron、SGLang：

```bash
(
  cd .deps/SeeNav-Agent/verl-agent
  USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
)
python -m pip install -r .deps/SeeNav-Agent/verl-agent/requirements.txt ai2thor==5.0.0 gym==0.26.2
python -m pip check
python -c 'import torch, transformers, vllm, flash_attn, ray, ai2thor; print(torch.__version__, torch.version.cuda)'
```

作者安装脚本指定 PyTorch 2.6.0、vLLM 0.8.5.post1 及 FlashAttention 2.7.4.post1；
SeeNav 依赖清单指定 Transformers 4.51.1。AI2-THOR 5.0.0 和 Gym 0.26.2 来自官方
[EmbodiedBench-Navigation 环境](https://github.com/EmbodiedBench/EmbodiedBench/blob/master/conda_envs/environment_eb-nav.yaml)。
该公开 checkout 的 `verl-agent` 包装文件引用了未随附的 README，运行器直接从原始源码目录加载，
不需要编辑上游包或 `pip install -e`。不把仓库的未锁定依赖称为已知的作者训练镜像。

## 检查并启动

```bash
python reproduce.py check
python reproduce.py plan
python -m unittest discover -s tests -v

# 保留原版 console + W&B；无线上账号时可由用户选择离线记录：
# export WANDB_MODE=offline
# 按实际获配的 GPU 编号填写；不要选其他任务的卡。
python reproduce.py train --gpus 0,1,2,3 --output runs/ebnav-vp-srgpo-opsd-4gpu
```

调度器已设 `CUDA_VISIBLE_DEVICES` 为 4 张卡时可省略 `--gpus`。输出目录须不存在且不含空格。
`plan` 分别显示纯一行四卡 diff 和加入 OPSD 开关后的实际入口。训练沿用原预处理器：
Geometry3K 只用于构造 8/8 行视觉加载器输入；**导航的场景、目标和训练轨迹来自 AI2-THOR 环境**，
不是在数学数据上训练导航。没有重新划分或替换上游的任务。

在另一个终端运行 `tail -f runs/ebnav-vp-srgpo-opsd-4gpu/train.log` 查看训练。运行目录保留四卡
launcher、实际 launcher、源码与补丁、技能和损失的版本/哈希记录、`train.log`、原生 `metrics/`
和 `checkpoints/`。
没有自动删除 checkpoint 或自动发布数据；退出码为 0 仅表示原生进程退出成功，正式验收仍需确认
末步 metrics、FSDP model/optimizer/extra 各 4 个 shard、dataloader 状态及 HF 导出文件。

## 上游原样保留的限制

- 论文还报告了 VLN-CE (R2R)；本仓库固定的公开源码提供 EB-Nav 的 SRGPO 入口，未找到相应的
  VLN-CE 训练/评测入口。本仓库不添加替代实现，也不宣称复现该表列。
- 原版训练 YAML 的相机俯仰为 90°，论文附录和独立评测的相机俯仰为 30°；
  论文正文与附录提示词也存在拼接顺序描述差异。实际双图代码按 `BEV | FV`。这些均不改。
- 原训练内验证是从 base 任务中取 8 个，不是独立评测的固定 60 任务成功率。作者独立评测
  `SeeNav/testEBNav.py` 引用了没有随这一 commit 发布的 `DualViewGrounding`、`ActionMarker`
  顶层模块。保留原版，不补写或替换评测逻辑。作者发布的 EB-Nav 权重见
  [官方 checkpoint](https://huggingface.co/wangzc9865/SeeNav-Agent)。

CPU 检查覆盖原版参数对照、补丁边界、Qwen 多模态位置、Teacher token/图像/掩码一致性，
以及 SDAR 梯度。它不代表 AI2-THOR 实机、四卡训练或论文结果已经验证。
代码分别遵循上游许可证；本仓库附带 SDAR 的许可证 [副本](licenses/SDAR.txt)。
