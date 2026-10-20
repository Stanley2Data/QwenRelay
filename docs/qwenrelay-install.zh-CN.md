# 安装 QwenRelay 0.1

[English](qwenrelay-install.md) · [README](../README.zh-CN.md)

## 准备机器

配置面向原生 Linux x86-64、Python 3.12、RTX 3090 24 GB。默认单卡 TP1 / AutoRound-fast；`--tp 2` / `--tp 4` 自动选择 EfficientThink。详见[配置与合并验证说明](qwenrelay-tp-profiles.md)。既有对测驱动为 580.126.20，用户态 CUDA 13.0；驱动需要兼容该运行库。`libcuda.so.1` 由宿主机驱动及容器运行时提供，本仓库不安装驱动。Docker 还需要已经配置好的 NVIDIA Container Toolkit。

建议预留至少 64 GiB 可用主机内存，以容纳加载及 32 GiB CPU 缓存预算；编译时留更多余量。磁盘约预留 60 GiB，用于约 21 GiB 模型、运行库、源码和编译缓存。这是资源规划建议，不是测得的最低要求。GPU KV 预算分别为 TP1 的 6 GiB、TP2 每卡 11,184,101,204 bytes、TP4 每卡 14 GiB。

## 安装运行库

系统需有 Python 3.12 的 venv/dev 支持、C/C++ 编译器、Git、GNU patch、curl 和证书。进入本仓库：

```bash
bash scripts/install-qwenrelay.sh
```

脚本只接受全新 `.venv`，按 `requirements-qwenrelay.txt` 安装直接依赖，获取固定 vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`，重建最终 F，再把源码覆盖到新 venv 并保持 wheel 原生扩展不变。不会下载权重、启动模型或改变已有服务。

可选环境变量为 `PYTHON`、`QWENRELAY_VENV`、`QWENRELAY_BUILD`、`VLLM_SOURCE`。最后一项可指定已经准备好的干净 vLLM 源码，构建输出目录必须是新的。直接依赖和源码有固定版本，传递依赖及 apt 包仍按当时解析；它不是完全封闭的依赖锁或逐字节相同的 native 构建。复现时保存 `pip freeze` 和原生文件 SHA。

安装依赖前先检查 Linux x86-64、Python 3.12、Git、GNU patch、C/C++ 工具及路径占用。
安装中断后选择新的 venv/build 目录；脚本不接续或删除半成品环境。

### 自定义目录

如果需要更换默认位置，用下面的流程代替默认安装，在同一终端中执行：

```bash
export QWENRELAY_VENV="$HOME/.local/share/qwenrelay/venv"
export QWENRELAY_BUILD="$HOME/.cache/qwenrelay/build"
export MODELS_ROOT="$HOME/models/qwenrelay"
export HF_HOME="$HOME/.cache/qwenrelay/huggingface"
bash scripts/install-qwenrelay.sh
"$QWENRELAY_VENV/bin/python" scripts/qwenrelay.py fetch-models --tp 1
bash scripts/start-qwenrelay.sh --tp 1 --dry-run
```

实际启动前按下文创建 API key。后续命令中的 `.venv/bin/python` 相应替换为
`"$QWENRELAY_VENV/bin/python"`。启动器依次选择显式 `QWENRELAY_PYTHON`、
`$QWENRELAY_VENV/bin/python`、仓库 `.venv/bin/python`。新终端中需重新 export，
建议使用绝对路径。`fetch-models` / `check-models --root` 和 `serve --models-root`
优先于 `MODELS_ROOT`；省略路径时三者共用该变量。

## 准备权重

```bash
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models
```

默认从 HyperQwen 使用的公开仓库下载固定版本输入，在 CPU 上准备 embedding，
再按 `locks/models-tp1.json` 校验测试时的完整最终文件哈希。公开来源为
[dbirks 基础模型](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/tree/1f05c441c4e64ae0549de44fa9ea5a6d43610314)、
[syvai fast 组件](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant/tree/124c14e7e8c7d2f5402933b9af368e772a9fcf0c) 和
[syvai DFlash2 W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16/tree/4d30ec736ffc6b8688dc2ae2b502d9b48bdec279)。
已有 AutoRound-fast 目录只检查、不原地修改；可复用 Hugging Face 下载缓存。
本轮打包尚未重新执行完整大文件准备，哈希不匹配会停止，不会放宽锁定条件。

Hub 缓存与最终模型分别占用磁盘；`HF_HOME` 可以指定缓存位置。CPU embedding 准备还需要
临时输入输出空间。失败的 `.qwenrelay-tp1-*` 暂存目录会保留供排查，重试前检查剩余空间。

双卡、四卡使用原来的 `locks/models.json` 和
[EfficientThink SimPO](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2/tree/882e1af3b8f5844f3b3163b132e343c06e91908f/NVFP4/AWQ-W4A16)，
草稿仍为同一个 DFlash2。EfficientThink 为 367 个 W4 非对称 +33 个 W8 对称 target 的混合 AWQ。
三套配置激活均为 BF16、KV 为 INT8，不切换为 FP8 草稿。权重遵循各模型作者许可，不进入 Git；
仓库只包含必要文本元数据、固定服务聊天模板和源码，仍使用 language-model-only。

```bash
# 显式 TP1 与不传 --tp 一致。
.venv/bin/python scripts/qwenrelay.py check-models --tp 1 --root /绝对路径/models
# EfficientThink 只需下载一次，供双卡、四卡使用。
.venv/bin/python scripts/qwenrelay.py fetch-models --tp 2 --root ./models
.venv/bin/python scripts/qwenrelay.py check-models --tp 4 --root ./models
```

单卡目录为 `Qwen3.8-27B-W4A16-AutoRound-fast/`，双卡、四卡目录为
`Qwen3.8-27B-EfficientThink-AWQ-W4A16/`，共同草稿目录为 `Qwen3.8-27B-DFlash2-W4A16/`。
这些目录可以共存。每次启动都完整检查所选资产的 SHA，较大权重需要一定时间。
准备细节见[TP 配置说明](qwenrelay-tp-profiles.md#weight-preparation)。

## 查看计划与启动

```bash
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt
bash scripts/start-qwenrelay.sh --devices 0 --dry-run
bash scripts/start-qwenrelay.sh --devices 0
```

默认 `127.0.0.1:18020/v1`、模型名 `qwenrelay`。设备、模型目录、端口、绑定地址和 key 文件可用参数指定；环境变量见 `.env.example`。原生脚本不把 `.env` 当 Shell 执行，请显式 export 或传参。`--dry-run` 仅打印计划，不检查模型/运行库，也不触碰 GPU，不能代表部署成功。

三套数值配置为 `config/single-3090.json`、`dual-3090.json`、`quad-3090.json`，由 `--tp 1/2/4` 选择，默认 TP1。双卡启动传 `--tp 2 --devices 0,1`，四卡传 `--tp 4 --devices 0,1,2,3`。`INT8_ACT` 和 Marlin INT8 输入开关会在启动环境中清除，避免继承父进程开关。32 GiB 是主机缓存总字节预算；TP1 单请求上限为 132,096 tokens，TP2/TP4 为 256K，不保证四条最大长度会话同驻留。

## 常用启动参数

先按 `--tp` / `--profile` 读取默认配置，再应用显式启动参数。省略的参数保留所选 TP 的默认值，
不会改写 JSON 配置文件；主模型仍按 TP1 → AutoRound-fast、TP2/TP4 → EfficientThink 选择。

| 参数 | 可用别名 | 单位 / 作用 |
|---|---|---|
| `--max-num-seqs` | `--concurrency` | 最大运行请求数，正整数；不是客户端排队数。默认 4。 |
| `--max-model-len` | `--context-length` | 每请求总上下文 token 上限，包含输入与输出；必须大于推测深度。 |
| `--draft-tokens` | `--num-speculative-tokens` | DFlash 推测验证深度，整数 1–63；默认 4。 |
| `--kv-cache-memory` | — | 每个 TP rank 的 GPU KV 预算；接受完整 bytes、`5GiB`、`512MiB`、`3.5GiB`，不接受含义不明确的 `5G` 或 `5GB`。 |
| `--kv-offloading-size` | `--cpu-kv-gib` | 整个 TP 组的 CPU KV 预算，单位 GiB，可用小数；默认 32，`0` 关闭 CPU 卸载及 O4，保留 GPU 前缀缓存。 |
| `--max-num-batched-tokens` | — | 每调度步的总 token 预算，正整数；默认 2,048。 |

GPU KV 预算必须为正的整数 bytes。CPU 预算必须为有限非负数；负数、NaN、Infinity 被拒绝。
调整并发 `C` 或推测深度 `K` 时，启动器自动设置：

```text
VLLM_SPEC_DECODE_ATTN_QMAX = K + 1
max_cudagraph_capture_size = C × (K + 1)
```

同时要求 `max-num-batched-tokens >= C × (K + 1)`。不满足时会在读取 key、加载模型或
访问 CUDA 前报错，提示增大批处理预算或减小并发/深度。QMAX 是逻辑请求长度，实际 scratch
还可能按 kernel tile 向上取整；Graph 上限不改变模型的训练草稿长度，也不增加 KV 缓存。
若推测深度超过草稿模型训练块，额外验证位置由现有 lookup 路径尝试补齐。

缓存页大小也随深度改变。非默认深度下启动器推导 `--block-size`，并把 20,800-token
保留间隔目标向上取整为整块的 `--prefix-cache-retention-interval`。深度 6 对应
816 / 21,216 tokens，默认深度 4 保持 800 / 20,800。可在 `effective_settings` 的
`hybrid_block_tokens` 和 `prefix_cache_retention_interval` 核对。这套推导仅针对锁定的
Qwen 主模型、INT8 PAD16 KV 与 FP16 循环状态。对外 GPU 预算参数转换为 vLLM 完整名称
`--kv-cache-memory-bytes`。

```bash
# TP1 默认值之上手动覆盖；此例逻辑 QMAX=7、Graph=14。
bash scripts/start-qwenrelay.sh --tp 1 \
  --max-num-seqs 2 --max-model-len 65536 --draft-tokens 6 \
  --kv-cache-memory 5GiB --kv-offloading-size 16 --dry-run

# 双卡，同样允许覆盖。省略的缓存预算等参数仍使用 TP2 默认值。
bash scripts/start-qwenrelay.sh --tp 2 --devices 0,1 \
  --concurrency 8 --context-length 65536 --draft-tokens 7 \
  --max-num-batched-tokens 1024 --dry-run

# 关闭 CPU 卸载，其他设置保留 TP1 默认值。
bash scripts/start-qwenrelay.sh --kv-offloading-size 0 --dry-run
bash scripts/start-qwenrelay.sh --help
```

核对 dry-run 输出的 `overrides`、`effective_settings`、`environment` 和最终 `argv`。
确认配置后移除 `--dry-run` 才会实际启动。调大上下文不等于增加物理缓存，调大并发/深度会
增加工作缓冲区及 Graph 开销；运行时仍检查模型上下文限制和实际可用显存。手动设置不继承
默认配置的性能验证结论。O5 目前优化最多 4 条请求、最多 2,048 query tokens 的合格预填充；
超出条件保留原生回退，并非所有调大后的形状都获得相同加速。

Docker 使用同一个启动器，可在 Compose 服务的 `command` 列表中追加这些参数，例如
`['--tp', '1', '--max-num-seqs', '2', '--draft-tokens', '6']`；修改后重新创建容器生效。
原生脚本不读取 `.env`，这些调参项通过命令行传入。

## DFlash 启动开关

当前 `single-3090`、`dual-3090`、`quad-3090` 三套配置均显式包含：

```ini
VLLM_SPEC_DECODE_ATTN=1
VLLM_SPEC_DECODE_ATTN_QMAX=5
VLLM_DFLASH2_LOOKUP=1
```

它们配置的是本项目修补后的 vLLM 运行时，与指定 DFlash 方法和草稿模型的 `--speculative-config` 配合使用。早期独立启动入口曾漏继承这些设置：DFlash 本身仍能运行，但长上下文性能明显回退。因此，仅检查源码补丁、模型 SHA 或服务是否启动，不足以确认运行配置等价。

| 设置 | 当前深度 4 配置中的作用 |
| --- | --- |
| `VLLM_SPEC_DECODE_ATTN=1` | 启用符合条件的 split-KV 推测验证 Attention 路径；漏配时，验证会绕过这项优化。 |
| `VLLM_SPEC_DECODE_ATTN_QMAX=5` | 设置逻辑上的每请求 query 上限，当前为 4 个草稿 token＋1 个验证 token。GQA=6 的实际缓冲区 query tile 为 10，启动时按实际最大调度请求数分配。缓冲区地址稳定关系到 CUDA Graph 捕获；它不是上下文长度，也不是所有并发请求的 query 总数。 |
| `VLLM_DFLASH2_LOOKUP=1` | 启用基于请求上下文的 lookup 辅助草稿候选生成；漏配会关闭这项辅助，不是关闭整套 DFlash。 |

两个布尔开关必须使用字面值 `1`，此运行时不会把 `true` 当作启用。上表的 `5` 是默认深度 4 时的值；使用 `--draft-tokens` 时 QMAX 与 Graph 上限会自动联动。绕过启动器时需自行保持它们一致，并重新验证。

`scripts/start-qwenrelay.sh` 和 `scripts/qwenrelay.py serve` 会自动注入有效配置，无需额外 export。显式命令行参数覆盖 profile 默认值，再覆盖父进程同名环境变量；常用调整通过上述参数完成，不能依赖外部 export 修改 QMAX。项目 Docker 路径也使用同一启动器。

若绕过启动器直接调用修补后的 vLLM，应在启动 Python 前，在同一个 Shell 中 export 这三项。自定义 systemd 单元直接启动 vLLM 时可使用：

```ini
[Service]
Environment="VLLM_SPEC_DECODE_ATTN=1"
Environment="VLLM_SPEC_DECODE_ATTN_QMAX=5"
Environment="VLLM_DFLASH2_LOOKUP=1"
```

修改单元后需执行 `systemctl daemon-reload` 并重启对应服务；修改文件或另一个 Shell 的变量，不会改变已运行进程。只把这三行写进 `.env` 也不会配置原生启动器，它不读取该文件。

正常启动前先查看计划：

```bash
bash scripts/start-qwenrelay.sh --devices 0 --dry-run
```

核对输出 `environment` 中的三项值，以及 `argv` 中推测配置的 `method: dflash`、所选 `num_speculative_tokens`（默认 4）、`draft_sample_method: probabilistic` 和正确的 DFlash2 模型路径。手动覆盖时也核对 `effective_settings` 中 QMAX 与 Graph 上限。四卡使用 `--profile quad-3090 --devices 0,1,2,3`。dry-run 仅确认计划；真实启动后，还应核对进程环境、启动配置，并在生成请求后查看 `/metrics` 中非零的 `spec_decode` 计数。草稿计数证明推测解码有实际活动，不能单独证明所有请求形状都走了 split-KV 路径。

## Docker
仓库提供使用同一启动器和配置的**本地构建方案**。宿主机不必先安装 Python/vLLM，可以用构建好的镜像准备权重和 key。在仓库目录中执行：

```bash
docker build -t qwenrelay:local .
mkdir -p models
cp .env.example .env
# 编辑 .env，将 MODEL_ASSET_DIR 设为主机模型目录的绝对路径。
# 单卡：TP=1，CUDA_VISIBLE_DEVICES=0
# 双卡：TP=2，CUDA_VISIBLE_DEVICES=0,1
# 四卡：TP=4，CUDA_VISIBLE_DEVICES=0,1,2,3

# 仅用 CPU 准备；--tp 与 .env 对应，2 和 4 共用权重。
docker run --rm --entrypoint /app/.venv/bin/python \
  --mount "type=bind,src=$PWD/models,dst=/models" \
  --mount type=volume,src=qwenrelay-download-cache,dst=/cache \
  qwenrelay:local scripts/qwenrelay.py fetch-models --tp 1

# 只创建一次，保留已有 key。
umask 077
if [ ! -e api_key.txt ]; then
  docker run --rm --entrypoint /app/.venv/bin/python qwenrelay:local \
    -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt
fi
docker compose up -d
docker compose logs -f qwenrelay
```

Compose 将权重只读挂载、key 作为 secret 挂载，使用独立卷保留编译缓存，并配置 40 GiB 共享内存。主机 API 端口绑定到 `127.0.0.1`。需要覆盖调参项时，修改服务的 `command` 列表并重新创建容器。

准备容器对模型目录使用**读写挂载**，服务容器使用**只读挂载**。O4 的 CPU 缓存分配在锁页主机内存中，不是一个 32 GiB 的 `/dev/shm` 文件；共享内存额度用于容器通信等需求，并非额外的物理内存。等待 `/v1/models` 成功后，沿用原生部署的请求示例即可。

当前没有 QwenRelay 预构建镜像，这份构建方案的全新构建与 GPU 启动仍待验证。见 [验证边界](qwenrelay-validation.md)。

## 离线重建

不需要导入 vLLM/PyTorch；准备 Python、Bash/GNU patch 与干净 vLLM 固定源码：

```bash
python3 scripts/qwenrelay.py check-pack
python3 scripts/qwenrelay.py rebuild --vllm-source /path/to/vllm --target /新目录/final-F
python3 -m unittest discover -s tests -v
```

源码只对 UTF-8 文本做 CRLF→LF 规范化；二进制和模型资产不变。重建顺序为 54 项上游补丁、3 项本地移植补丁、15 文件最终覆盖层，完整核对 3,208 份规范化输出的 SHA 与 Python 语法。底座保留历史原始 SHA 与换行规范；后续 TP1/TP4 合并的代码差异、前后身份和验证范围另有记录，见 `locks/source.json` 与 TP 配置说明。另两项 review 补丁仅供阅读，不要再应用。输出旁保存检查报告；已有目录和错误底座均拒绝。
