# DynamicCodec

基于 [argbind](https://github.com/pseeth/argbind) + ClassChoice 的动态音频编解码框架。把 encoder / decoder / quantizer / vocoder 拆成可插拔组件，改一行 YAML 就能换组合（例如 `dac+rvq`、`mel+vocos`、`encodec+vq`），不用动模型代码。

- **动态组件组合**：`conf/base.yaml` 选 `encoder/decoder/quantizer/vocoder` → 配置自动按 `conf/model/<kind>/<name>.yaml` 拉取
- **多输入格式**：`wav` / `melspec` / `repr`（SSL 表征）
- **训练 + 推理一套配置**：通过 `state: train | inference` 切换
- **Docker 一键编排**：内网 / 公网两条路径，幂等可复跑
- **VCTK manifest 工作流**：CSV 索引训练数据，避免反复扫盘

---

## 快速开始

### 1. 拉代码 + 起容器

```bash
git clone https://github.com/luki0810/DynamicCodec.git
cd DynamicCodec

# 一条命令搞定：检查 docker、拉/构建镜像、创建容器、装 pip 包、打补丁、跑自检
bash scripts/setup_container.sh
```

镜像选择是自动的：如果设置了 `DYC_INTERNAL_IMAGE` 环境变量并且能拉到该镜像，就直接用；否则 fallback 到本地构建 `dynamiccodec:local`（基于 Docker Hub 的 `pytorch/pytorch:2.6.0-cuda12.4`，首次约 10 分钟）。

```bash
# 可选：指定内网/私有镜像
export DYC_INTERNAL_IMAGE=registry.example.com/team/dyc:v1
# 可选：把宿主机上的数据盘挂进容器
export DYC_DATA_MOUNT=/path/to/data
```

显式控制：

```bash
bash scripts/setup_container.sh --public         # 强制走公网构建
bash scripts/setup_container.sh --image my:tag   # 用自己准备的镜像
bash scripts/setup_container.sh --rm             # 销毁旧容器后重建
bash scripts/setup_container.sh --help           # 看完整说明
```

容器名固定为 `dyc_dev`，所有命令都通过 `docker exec` 进入：

```bash
sudo docker exec -it dyc_dev bash       # 交互式
sudo docker exec dyc_dev <command>      # 单条命令
```

### 2. 推理：用预置的 DAC checkpoint

仓库里 `runs/dac-result/best/` 自带一份 48 kHz / 8-codebook DAC 权重，开箱即用。

先确认 `conf/base.yaml` 里 `state: inference`（训练后会被切到 `train`，推理前切回来）：

```yaml
state: inference
```

跑推理：

```bash
sudo docker exec dyc_dev bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/inference_dac --args.debug 1"
```

默认读 `wav_file/input_wav/p226_002.wav`，输出 `runs/inference_dac/recon.wav`。

换 checkpoint：改 `conf/inference.yaml` 的 `exp_name` / `tag` 即可。

### 3. 切换组件：改一行 YAML

`conf/base.yaml` 是组件总开关：

```yaml
# 组件选择
state: inference     # train / inference
input_format: wav    # wav / melspec / repr
encoder: dac         # dac / encodec / mel / repcodec
quantizer: rvq       # rvq / vq / bsq / fsq
decoder: dac         # dac / encodec / mel
vocoder: null        # null（不用）/ vocos
```

每个字段对应 `conf/model/<kind>/<name>.yaml`，会被自动 include 进来。新组合即改即用，不需要改代码。

> **注意**：`vocoder` 不用时一定要写 `null`（YAML 关键字），不要写 `None`（会被解析成字符串触发 `class_choices` 里的判断 bug）。

---

## 训练

### 准备 VCTK manifest

`audiotools.AudioLoader` 接受两种数据源：folder（递归扫盘）或 CSV manifest（带 `path` 列）。本项目用 manifest，避免训练时反复扫 CFS。

```bash
# host 上跑（不需要进容器）
python scripts/build_vctk_manifest.py
```

输出：

| 文件 | 行数 | 说明 |
|------|------|------|
| `data/manifests/vctk/train.csv` | ~42350 | 96 % |
| `data/manifests/vctk/val.csv`   | ~947   | 2 % |
| `data/manifests/vctk/test.csv`  | ~945   | 2 % |

切分逻辑：所有 109 个 speaker 都参与训练（speaker-dependent），每个 speaker 内部按文件名排序后按 `idx % 50` 切（确定性，可复现）。

> **VCTK 路径**：通过环境变量 `VCTK_ROOT` 指向本地 VCTK `wav48` 目录，或直接修改 `scripts/build_vctk_manifest.py` 里的 `VCTK_ROOT` 默认值。
>
> ```bash
> VCTK_ROOT=/path/to/vctk/wav48 python scripts/build_vctk_manifest.py
> ```

`conf/train/dataset.yaml` 已经指向这三个 CSV 的相对路径。

### 启动训练

切到训练状态：`conf/base.yaml` 设 `state: train`。

```bash
sudo docker exec dyc_dev bash -c \
  "cd /app && python train.py --conf_path conf/base.yaml --save_path runs/dac_run --args.debug 1"
```

主要训练参数都在 `conf/train.yaml` 和 `conf/train/*.yaml` 里：

| 文件 | 控制 |
|------|------|
| `conf/train.yaml` | `num_iters` / `save_iters` / `valid_freq` / `batch_size` / `lambdas` / `Accelerator.amp` / 是否 resume |
| `conf/train/adam.yaml` | 优化器 + 学习率 |
| `conf/train/dataset.yaml` | 数据 manifest + 训练/验证/测试切片长度 + 数据增强 |
| `conf/train/discriminator.yaml` | GAN 判别器配置 |
| `conf/train/loss.yaml` | Mel / STFT / SISDR 等损失参数 |

输出结构（`runs/<save_path>/`）：

```
args.yaml                 # 完整运行参数快照
log.txt                   # 文本训练日志
logs/<timestamp>/         # TensorBoard events
best/                     # val loss 最低时的 checkpoint
  ├── dynamiccodec/       {weights, optimizer, scheduler, tracker, metadata}.pth
  └── dynamicdiscriminator/
latest/                   # 最新 checkpoint（同上结构）
```

### 多 GPU 训练

```bash
sudo docker exec dyc_dev bash -c \
  "cd /app && torchrun --nproc_per_node gpu train.py --conf_path conf/base.yaml --save_path runs/dac_run --args.debug 1"
```

### 续训

`conf/train.yaml` 里：

```yaml
load.resume: true
load.tag: best        # 或 latest
load.only_load_weights: true   # 只加载权重；false 则把 optimizer / scheduler / tracker 也恢复
```

resume 路径 = `<save_path>/<tag>/`。

---

## 加入新组件

以新增 encoder 为例（decoder / quantizer 类似）：

1. **写模型代码**：在 `model/encoder/` 下加文件，类继承 `AbsEncoder`（同理 `AbsDecoder` / `AbsQuantizer`）
2. **注册组件**：在 `model/all_choices.py` 的对应注册表里加一行
3. **加配置**：在 `conf/model/encoder/<your_name>.yaml` 里写超参
4. **切换使用**：`conf/base.yaml` 把 `encoder: dac` 改成 `encoder: <your_name>`

整个过程零侵入，不需要改 `train.py` / `main.py`。

---

## 支持的组件

| 类别 | 选项 |
|------|------|
| **Encoder**     | `dac`, `encodec`, `mel`, `repcodec`(SSL) |
| **Decoder**     | `dac`, `encodec`, `mel` |
| **Quantizer**   | `rvq`（残差 VQ）, `vq`, `bsq`, `fsq`（Finite Scalar Quantization） |
| **Vocoder**     | `vocos`, `null`（不接 vocoder） |
| **Input format**| `wav`, `melspec`, `repr`（SSL 特征） |

仓库里的 `cosmos` 编/解码器是 NVIDIA Cosmos 图像 tokenizer（2D Conv），未适配 1D 音频，目前没注册到选择表里；代码保留在 `model/{encoder,decoder}/cosmos.py` 供未来 2D 实验用。

`repr` + `repcodec` 走的是 SSL 特征路线，需要预下载 SSL checkpoint 到 `ckpt/<model_type>/...`（默认 `ckpt/data2vec/base_no_ft.pt`），并在容器里装好 `fairseq` / `openai-whisper`（`scripts/setup_container.sh` 已包含）。

> 想确认所有组件是否健康，跑一遍 `python scripts/probe_components.py`：每个组合在独立子进程里随机初始化、跑一次 forward，几秒内出结果。

---

## 项目结构

```
DynamicCodec/
├── .docker/                 # 公网 Dockerfile（pytorch/pytorch 基础镜像）
├── conf/
│   ├── base.yaml            # 主配置（state + 组件选择）
│   ├── inference.yaml       # 推理时 checkpoint 加载
│   ├── train.yaml           # 训练超参
│   ├── input/               # 输入格式配置（wav / melspec / repr）
│   ├── model/               # 各组件配置
│   └── train/               # 训练子配置（adam / dataset / discriminator / loss）
├── data/                    # 数据加载模块
│   ├── ssl/                 # SSL 特征提取（HuBERT / Whisper / Data2Vec）
│   ├── melspec.py           # Mel 频谱特征
│   ├── repr.py              # SSL 表征
│   └── manifests/           # 训练 manifest CSV（脚本生成）
├── model/
│   ├── encoder/             # Encoder 实现
│   ├── decoder/
│   ├── quantizer/
│   ├── vocoder/
│   ├── nn/                  # 神经网络层 / Loss / Discriminator backbones
│   ├── utils/               # ClassChoice、动态 argbind 加载、CodecMixin、logger
│   ├── all_choices.py       # 组件注册表
│   └── build.py             # DynamicTask / DynamicCodec / DynamicDiscriminator
├── scripts/
│   ├── setup_container.sh             # 容器一键编排
│   ├── build_vctk_manifest.py         # 生成 train/val/test CSV
│   ├── patch_argbind.py               # argbind argv 兼容补丁
│   ├── patch_audiotools_torchload.py  # PyTorch 2.6 weights_only 兼容补丁
│   └── smoke_check_container.py       # 容器内自检
├── runs/                    # 实验输出（gitignore；自带 dac-result/best 预训练权重）
├── wav_file/                # 推理输入 / 输出示例
├── main.py                  # 推理入口
├── train.py                 # 训练入口
└── README.md
```

---

## Docker 细节

`scripts/setup_container.sh` 做的事（幂等，再跑一次会跳过已完成的步骤）：

1. 检查 docker / sudo 权限；首次把当前用户加入 `docker` 组（下次登录生效）
2. 选定镜像（必要时 build 公网 `dynamiccodec:local`）
3. 复用或创建 `dyc_dev` 容器（挂载项目；如果设置了 `DYC_DATA_MOUNT` 会一并挂载）
4. 装缺失的 pip 包：`descript-audiotools 0.7.2`、`argbind 0.3.9`、`vocos 0.1.0`、`typeguard`、`humanfriendly`
5. 打 `argbind` 补丁（`parse_args(argv=...)` 支持显式 argv）
6. 打 `audiotools` 补丁（兼容 PyTorch 2.6 默认 `weights_only=True`）
7. 跑 `scripts/smoke_check_container.py` 验证补丁与 manifest
8. 把过去 root 进程写到 `runs/*` 的文件 chown 回当前用户

容器生命周期管理：

```bash
sudo docker stop dyc_dev        # 停（保留状态）
sudo docker start dyc_dev       # 重启
bash scripts/setup_container.sh --rm   # 完全重置（pip + patch 会被脚本自动恢复）
```

单独构建公网镜像（不创建容器）：

```bash
bash .docker/build.sh                    # 默认 tag: dynamiccodec:local
bash .docker/build.sh dynamiccodec:dev   # 自定义 tag
```

`.docker/Dockerfile` 不 COPY 代码，运行时用 `-v` 挂载，所以镜像可跨分支复用。

---

## 许可证

MIT，详见 `LICENSE`。

## 贡献

欢迎提 Issue / PR。
