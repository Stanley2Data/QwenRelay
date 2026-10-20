# QwenRelay

**面向 RTX 3090 的 Qwen 推理优化：长上下文、小并发，以及可恢复复用的会话缓存。**

[English](README.md) · [安装文档](docs/qwenrelay-install.zh-CN.md) · [性能与口径](docs/qwenrelay-benchmarks.md) · [上游和归属](UPSTREAM.md)

QwenRelay 基于 [HyperQwen](https://github.com/syv-ai/HyperQwen)，固定 vLLM 0.31.0。默认使用 **单张 RTX 3090 24 GB、TP1、AutoRound-fast、DFlash2 深度 4**；`--tp 2` / `--tp 4` 自动切换为 EfficientThink 及对应优化配置，包含 E2 对齐 INT8 KV、O4 主机历史缓存、PF2/O2/O5 预填充优化、CPU 缓存恢复和四路 CUDA Graph 覆盖修复。

当前是首版源码候选。底层 TP2 F 已有 GPU 性能及缓存恢复验证；**当前 TP1/TP2/TP4 合并源码完成离线检查，完整权重准备、全新 Docker 构建及合并后的 GPU 验证仍待执行。** 详见[验证范围](docs/qwenrelay-validation.md)。

## 默认配置

| 项目 | 配置 |
|---|---|
| 硬件与系统 | 原生 Linux、Python 3.12、单张 RTX 3090 24 GB |
| 推理底座 | vLLM 0.31.0、PyTorch 2.13.0 / CUDA 13.0、Triton 3.7.1 |
| 主模型 | 固定 HyperQwen AutoRound-fast W4A16 |
| 草稿 | 固定 syvai DFlash2 W4A16，4 个 speculative tokens |
| 调度 | 最多 4 条运行序列，单步最多 2,048 tokens |
| 上下文 | 单请求配置上限 132,096 tokens，实际受缓存容量约束 |
| 缓存 | INT8 per-token/head KV，FP16 循环状态，32 GiB CPU 卸载预算 |
| 激活 | BF16，关闭 `INT8_ACT` |

上下文上限受 6 GiB GPU KV 预算约束，不代表四条最大长度会话能同时驻留。视觉、任意其他权重或超高并发暂不在这些配置的范围内。

## 快速开始

```bash
git clone https://github.com/Stanley2Data/QwenRelay.git
cd QwenRelay
```

进入仓库目录；准备兼容的 NVIDIA 驱动、Linux Python 3.12、Git 和 GNU patch：

```bash
# 创建全新 .venv，安装依赖并重建固定源码；不会启动模型。
bash scripts/install-qwenrelay.sh

# 下载固定版本的上游 AutoRound-fast 输入，在 CPU 上准备 embedding，
# 再校验最终权重哈希；Git 不包含权重。
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models

umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt

# 只查看启动计划，不加载模型、不探测 GPU。
bash scripts/start-qwenrelay.sh --devices 0 --dry-run

# 完整检查权重及运行库身份后启动。
bash scripts/start-qwenrelay.sh --devices 0
```

默认接口为 `http://127.0.0.1:18020/v1`，模型名 `qwenrelay`，API key 从本地 `api_key.txt` 读取。在 OpenAI 兼容客户端中填写这个地址和 key。详细驱动、内存、Docker 和已有权重接入方法见[中文安装文档](docs/qwenrelay-install.zh-CN.md)。

默认深度 4 时，启动器会自动注入以下 **DFlash 性能开关**；覆盖深度时会重算 QMAX：

```ini
VLLM_SPEC_DECODE_ATTN=1
VLLM_SPEC_DECODE_ATTN_QMAX=5
VLLM_DFLASH2_LOOKUP=1
```

早期独立启动入口曾漏继承这三项。DFlash 模型启动成功，不代表 split-KV 验证 Attention 和 lookup 草稿路径已启用。直接调用 vLLM 或编写自定义服务时也应保留它们；`5` 对应每请求“4 个草稿 token＋1 个验证 token”。用途、systemd 写法及核对方法见 [DFlash 启动开关](docs/qwenrelay-install.zh-CN.md#dflash-启动开关)。

按参数选择卡数，默认 TP1：

```bash
# 双卡、四卡共用 EfficientThink 权重，只需下载一次。
.venv/bin/python scripts/qwenrelay.py fetch-models --tp 2 --root ./models
bash scripts/start-qwenrelay.sh --tp 2 --devices 0,1
bash scripts/start-qwenrelay.sh --tp 4 --devices 0,1,2,3
```

| 启动参数 | 自动选择主模型 | 默认上下文上限 | 默认每卡 GPU KV |
|---|---|---:|---:|
| 不传 / `--tp 1` | AutoRound-fast | 132,096 | 6 GiB |
| `--tp 2` | EfficientThink | 262,144 | 11,184,101,204 bytes |
| `--tp 4` | EfficientThink | 262,144 | 14 GiB |

三套默认配置均使用 DFlash2、最多四路请求、32 GiB 整组 CPU 缓存，草稿 TP 与主模型一致。
保留原有 `--profile` 参数；同时传入时须与 `--tp` 一致。详见[配置、上游权重准备与合并说明](docs/qwenrelay-tp-profiles.md)和[历史 TP4 验证记录](docs/qwenrelay-tp4-validation.md)。

TP 选择默认配置，显式启动参数可以覆盖并发、上下文、推测深度和缓存预算：

```bash
bash scripts/start-qwenrelay.sh --tp 1 \
  --max-num-seqs 2 --max-model-len 65536 --draft-tokens 6 \
  --kv-cache-memory 5GiB --kv-offloading-size 16 --dry-run
```

此例自动设置逻辑 QMAX 为 `7`、Graph 捕获上限为 `2 × (6 + 1) = 14`。
GPU KV 预算按每卡计算，CPU KV 预算按整个 TP 组计算；`--kv-offloading-size 0` 关闭 CPU 卸载。
完整参数、别名与校验规则见[常用启动参数](docs/qwenrelay-install.zh-CN.md#常用启动参数)。
`--dry-run` 只查看有效配置；手动调参后的显存容量与性能仍需实际验证。

## 优化内容

- **E2**：INT8 KV head 按 16 字节对齐，并适配混合 Attention/GDN 页布局。
- **O4**：固定字节预算下划分完整 Attention、循环状态和草稿历史缓存，修复对齐交接及 CPU 恢复。设计参考 1Cat 的缓存讨论，来源见[第三方说明](THIRD_PARTY_NOTICES.md)。
- **PF2/O2/O5**：Prefill 复用 INT8 KV，覆盖限定形状的 TP1（24Q/4KV）、TP2（12Q/2KV）与 TP4（6Q/1KV），保留原生回退；TP1 的 O5 分两组复用工作区。O5 优化符合条件的变长混合预填充，纯 Decode 加速来自独立的 Graph 修复。
- **CUDA Graph**：默认捕获至 20 tokens，覆盖四请求 ×（4 草稿 + 1 验证）；覆盖并发或深度时自动重算上限，同时保留编译缓存身份及缓冲区生命周期修复。

Attention 临时缓冲区在初始化时按实际调度容量预分配；TP1 另启用经过加载检查的草稿输出头共享。源码身份见 [locks/source.json](locks/source.json)，增量补丁见 [patches/qwenrelay](patches/qwenrelay)。

## 实测表现

以下是本次合并前的 **TP2 EfficientThink 历史对测**。同一权重、GPU、原生运行库和缓存字节预算，每臂 1 暖轮 + 3 正式轮。下表是 **`C ÷ 平均 TPOT` 的三轮中位数**，不是总输出吞吐：

| 原项目测试场景 | H | F |
|---|---:|---:|
| 真实题 C4，temperature=0 | 113.19 tok/s | 374.92 tok/s |
| 真实题 C4，默认采样 | 107.50 tok/s | 360.11 tok/s |
| 合成 1K 输入 C4，T0，固定生成 1,024 tokens | 167.46 tok/s | 538.45 tok/s |

也保留回退：原 T0 C2 −2.63%、默认采样 C1 −1.85%；合成 1K C1 波动较大。128K C4 双方均没有合格的四路共同稳态生成片段，不能称为纯四路稳态 Decode。完整十档 Decode、总吞吐与方法限制见[性能文档](docs/qwenrelay-benchmarks.md)。

## 开发与来源

```bash
python3 scripts/qwenrelay.py check-pack
python3 -m unittest discover -s tests -v
```

本仓库源码采用 Apache-2.0，模型与依赖分别遵循各自许可。保留 HyperQwen、vLLM、KVarN 等原作者信息。上游的 batch、单卡脚本和[原始 README](README.hyperqwen.md)保留供追溯与开发，配置范围以本页和 `config/single-3090.json`、`dual-3090.json`、`quad-3090.json` 为准。参与修改请看 [CONTRIBUTING.md](CONTRIBUTING.md)。
