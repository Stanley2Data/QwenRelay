# 安装 QwenRelay 0.1

[English](qwenrelay-install.md) · [README](../README.zh-CN.md)

## 准备机器

首版支持原生 Linux x86-64、Python 3.12、双 RTX 3090 24 GB。既有对测驱动为 580.126.20，用户态 CUDA 13.0；驱动需要兼容该运行库。`libcuda.so.1` 由宿主机驱动及容器运行时提供，本仓库不安装驱动。Docker 还需要已经配置好的 NVIDIA Container Toolkit。

建议预留至少 64 GiB 可用主机内存，以容纳加载及 32 GiB CPU 缓存预算；编译时留更多余量。磁盘约预留 60 GiB，用于约 21 GiB 模型、运行库、源码和编译缓存。这是资源规划建议，不是测得的最低要求。每卡显式 KV 预算为 11,184,101,204 bytes。

## 安装运行库

系统需有 Python 3.12 的 venv/dev 支持、C/C++ 编译器、Git、GNU patch、curl 和证书。进入本仓库：

```bash
bash scripts/install-qwenrelay.sh
```

脚本只接受全新 `.venv`，按 `requirements-qwenrelay.txt` 安装直接依赖，获取固定 vLLM commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`，重建最终 F，再把源码覆盖到新 venv 并保持 wheel 原生扩展不变。不会下载权重、启动模型或改变已有服务。

可选环境变量为 `PYTHON`、`QWENRELAY_VENV`、`QWENRELAY_BUILD`、`VLLM_SOURCE`。最后一项可指定已经准备好的干净 vLLM 源码，构建输出目录必须是新的。直接依赖和源码有固定版本，传递依赖及 apt 包仍按当时解析；它不是完全封闭的依赖锁或逐字节相同的 native 构建。复现时保存 `pip freeze` 和原生文件 SHA。

## 准备权重

```bash
.venv/bin/python scripts/qwenrelay.py fetch-models --root ./models
```

这是显式联网下载动作，从原作者固定完整 commit 获取主模型和草稿，并验证每份文件完整 SHA256/大小。不会退回其他模型或覆盖不匹配文件。来源分别为：

- [EfficientThink SimPO](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2)，commit `882e1af3b8f5844f3b3163b132e343c06e91908f`，子目录 `NVFP4/AWQ-W4A16/`。这里选择的是 AWQ。
- [syvai DFlash2 W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)，commit `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`。

权重不进入 Git，分别遵循模型作者许可。主模型是 367 个 W4 非对称 +33 个 W8 对称 target 的混合 AWQ，激活保持 BF16；草稿为 GPTQ W4A16。AutoRound、Coder390、FP8 草稿与 INT8 激活均不是本次双卡基准配置。

仓库只包含测试时使用的文本模型索引与聊天模板等小型元数据。索引映射主权重和 BF16 视觉/MTP 文件，启动仍使用 language-model-only。已有完整模型可以直接检查：

```bash
.venv/bin/python scripts/qwenrelay.py check-models --root /绝对路径/models
```

其下应有 `Qwen3.8-27B-EfficientThink-AWQ-W4A16/` 与 `Qwen3.8-27B-DFlash2-W4A16/`。每次启动会完整读取并检查模型 SHA，约 21 GiB 需要一定时间。

## 查看计划与启动

```bash
umask 077
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > api_key.txt
bash scripts/start-qwenrelay.sh --devices 0,1 --dry-run
bash scripts/start-qwenrelay.sh --devices 0,1
```

默认 `127.0.0.1:18020/v1`、模型名 `qwenrelay`。设备、模型目录、端口、绑定地址和 key 文件可用参数指定；环境变量见 `.env.example`。原生脚本不把 `.env` 当 Shell 执行，请显式 export 或传参。`--dry-run` 仅打印计划，不检查模型/运行库，也不触碰 GPU，不能代表部署成功。

唯一数值配置是 `config/dual-3090.json`。`INT8_ACT` 和 Marlin INT8 输入开关会在启动环境中清除，避免继承父进程开关。32 GiB 是主机缓存总字节预算；256K 是单请求上限，不保证四条 256K 同驻留。

## Docker

根目录 Dockerfile 构建同一安装路径，没有借用上游预构建镜像。先准备权重和 `api_key.txt`，复制 `.env.example` 为 `.env`，把 `MODEL_ASSET_DIR` 改为主机上模型父目录的绝对路径：

```bash
docker compose build
docker compose up -d
docker compose logs -f qwenrelay
```

容器暴露 NVIDIA 设备，由 `CUDA_VISIBLE_DEVICES` 选择两卡（默认 `0,1`），主机端口仅发布到 loopback。权重只读挂载、key 作为 secret、编译缓存使用独立卷，`/dev/shm` 为 40 GiB。不需要 privileged 或宿主 Docker socket，也没有自动重启策略。

**本轮整理没有执行全新 Docker 构建或 GPU 启动。** 以前双卡运行库验证和本次公开安装路径检查分开记录，详见[验证边界](qwenrelay-validation.md)。

## 离线重建

不需要导入 vLLM/PyTorch；准备 Python、Bash/GNU patch 与干净 vLLM 固定源码：

```bash
python3 scripts/qwenrelay.py check-pack
python3 scripts/qwenrelay.py rebuild --vllm-source /path/to/vllm --target /新目录/final-F
python3 -m unittest discover -s tests -v
```

源码只对 UTF-8 文本做 CRLF→LF 规范化；二进制和模型资产不变。重建顺序为 54 项上游补丁、3 项本地移植补丁、6 文件最终缓存/Graph 覆盖层，完整核对 3,208 份规范化输出的 SHA 与 Python 语法。与旧测试工作树的区别仅为文本换行，原始 SHA 及规则保留在 `locks/source.json`。另两项 review 补丁仅供阅读，不要再应用。输出旁保存检查报告；已有目录和错误底座均拒绝。
