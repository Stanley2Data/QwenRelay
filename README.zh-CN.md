# QwenRelay

**面向双 RTX 3090 的 Qwen 推理优化：长上下文、小并发，以及可恢复复用的会话缓存。**

[English](README.md) · [安装文档](docs/qwenrelay-install.zh-CN.md) · [性能与口径](docs/qwenrelay-benchmarks.md) · [上游和归属](UPSTREAM.md)

QwenRelay 基于 [HyperQwen](https://github.com/syv-ai/HyperQwen)，固定 vLLM 0.31.0。首版支持已经实测的 **双 RTX 3090 24 GB、TP2、DFlash2 深度 4** 配置，包含 E2 对齐 INT8 KV、O4 主机历史缓存、PF2/O2/O5 预填充优化、CPU 缓存恢复和四路 CUDA Graph 覆盖修复。

当前是首版源码候选。底层 TP2 F 已有 GPU 性能及缓存恢复验证；**本次新整理的安装脚本和 Docker 路径完成离线检查，尚未重新执行全新 Docker 构建与模型启动。** 详见[验证范围](docs/qwenrelay-validation.md)。

## 首版配置

| 项目 | 配置 |
|---|---|
| 硬件与系统 | 原生 Linux、Python 3.12、双 RTX 3090 24 GB |
| 推理底座 | vLLM 0.31.0、PyTorch 2.13.0 / CUDA 13.0、Triton 3.7.1 |
| 主模型 | 固定 EfficientThink SimPO 混合 AWQ W4A16 |
| 草稿 | 固定 syvai DFlash2 W4A16，4 个 speculative tokens |
| 调度 | 最多 4 条运行序列，单步最多 2,048 tokens |
| 上下文 | 单请求配置上限 262,144 tokens，实际受缓存容量约束 |
| 缓存 | INT8 per-token/head KV，FP16 循环状态，32 GiB CPU 卸载预算 |
| 激活 | BF16，关闭 `INT8_ACT` |

256K 是单请求配置上限，不代表四条 256K 会话能同时驻留。首版暂不承诺单卡、视觉、其他权重或超高并发。

## 快速开始

```bash
git clone https://github.com/Stanley2Data/QwenRelay.git
cd QwenRelay
```

进入仓库目录；准备兼容的 NVIDIA 驱动、Linux Python 3.12、Git 和 GNU patch：

```bash
# 创建全新 .venv，安装依赖并重建固定源码；不会启动模型。
bash scripts/install-qwenrelay.sh

# 显式下载固定主模型和草稿，约 21 GiB；Git 不包含权重。
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models

umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt

# 只查看启动计划，不加载模型、不探测 GPU。
bash scripts/start-qwenrelay.sh --devices 0,1 --dry-run

# 完整检查权重及运行库身份后启动。
bash scripts/start-qwenrelay.sh --devices 0,1
```

默认接口为 `http://127.0.0.1:18020/v1`，模型名 `qwenrelay`，API key 从本地 `api_key.txt` 读取。在 OpenAI 兼容客户端中填写这个地址和 key。详细驱动、内存、Docker 和已有权重接入方法见[中文安装文档](docs/qwenrelay-install.zh-CN.md)。

## 优化内容

- **E2**：INT8 KV head 按 16 字节对齐，并适配混合 Attention/GDN 页布局。
- **O4**：固定字节预算下划分完整 Attention、循环状态和草稿历史缓存，修复对齐交接及 CPU 恢复。设计参考 1Cat 的缓存讨论，来源见[第三方说明](THIRD_PARTY_NOTICES.md)。
- **PF2/O2/O5**：Prefill 复用 INT8 KV，覆盖限定形状的 TP2 路径并保留原生回退。O5 优化符合条件的变长混合预填充，纯 Decode 加速来自独立的 Graph 修复。
- **CUDA Graph**：捕获至 20 tokens，覆盖四请求 ×（4 草稿 + 1 验证），同时适配编译缓存身份及缓冲区生命周期。

源码身份见 [locks/source.json](locks/source.json)，增量补丁见 [patches/qwenrelay](patches/qwenrelay)。

## 实测表现

同一权重、GPU、原生运行库和缓存字节预算，每臂 1 暖轮 + 3 正式轮。下表是 **`C ÷ 平均 TPOT` 的三轮中位数**，不是总输出吞吐：

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

本仓库源码采用 Apache-2.0，模型与依赖分别遵循各自许可。保留 HyperQwen、vLLM、KVarN 等原作者信息。上游的 batch、单卡脚本和[原始 README](README.hyperqwen.md)保留供追溯与开发，首版支持范围以本页和 `config/dual-3090.json` 为准。参与修改请看 [CONTRIBUTING.md](CONTRIBUTING.md)。
