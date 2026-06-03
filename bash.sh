#!/usr/bin/env bash
# DynamicCodec — component combinations, run reference.
#
# 原始用法是改 conf/base.yaml 的 5 个字段（state / input_format / encoder /
# quantizer / decoder / vocoder），然后跑一条 python 命令。下面每个组合给出
#   1) base.yaml 应该长什么样
#   2) 对应的单条命令（推理 / 训练）
#
# 直接把要跑的那一段命令复制出来执行即可。
# 容器名按需替换为 dyc_dev 或 dyc_luki（取决于你 setup_container.sh 用的是哪个版本）。

# ----- host-side env (optional) -----------------------------------------------
export HF_HOME=/path/to/huggingface
# source /path/to/anaconda3/bin/activate /path/to/anaconda3/envs/dynamic
# cd /path/to/DynamicCodec
# export PYTHONPATH="$PWD:$PYTHONPATH"

CTR=dyc_dev   # or dyc_luki for older setup

# ==============================================================================
# 1) dac-rvq           wav + dac + rvq + dac          (用 runs/dac-result/best)
# ==============================================================================
# conf/base.yaml:
#   state: inference
#   input_format: wav
#   encoder: dac
#   quantizer: rvq
#   decoder: dac
#   vocoder: null
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/dac_rvq --args.debug 1"

# ==============================================================================
# 2) dac-vq            wav + dac + vq + dac           (random init)
# ==============================================================================
# conf/base.yaml:
#   state: inference
#   input_format: wav
#   encoder: dac
#   quantizer: vq
#   decoder: dac
#   vocoder: null
# conf/inference.yaml: resume: false       # 没有 vq 的 ckpt，关掉 resume
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/dac_vq --args.debug 1"

# ==============================================================================
# 3) dac-bsq           wav + dac + bsq + dac          (random init)
# ==============================================================================
# conf/base.yaml:  encoder: dac    quantizer: bsq    decoder: dac
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/dac_bsq --args.debug 1"

# ==============================================================================
# 4) dac-fsq           wav + dac + fsq + dac          (random init)
# ==============================================================================
# conf/base.yaml:  encoder: dac    quantizer: fsq    decoder: dac
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/dac_fsq --args.debug 1"

# ==============================================================================
# 5) encodec-rvq       wav + encodec + rvq + encodec  (random init)
# ==============================================================================
# conf/base.yaml:
#   encoder: encodec
#   quantizer: rvq
#   decoder: encodec
#   (input_format: wav, vocoder: null)
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/encodec_rvq --args.debug 1"

# ==============================================================================
# 6) mel-rvq           melspec + mel + rvq + mel      (mel-domain reconstruction)
# ==============================================================================
# conf/base.yaml:
#   input_format: melspec
#   encoder: mel
#   quantizer: rvq
#   decoder: mel
#   vocoder: null
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/mel_rvq --args.debug 1"

# ==============================================================================
# 7) mel-rvq-vocos     melspec + mel + rvq + mel + vocos    (mel → vocoder → wav)
# ==============================================================================
# conf/base.yaml:
#   input_format: melspec
#   encoder: mel
#   quantizer: rvq
#   decoder: mel
#   vocoder: vocos
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/mel_rvq_vocos --args.debug 1"

# ==============================================================================
# 8a) repcodec-rvq-data2vec   repr + repcodec + rvq + dac   (SSL=data2vec, 768-dim)
# ==============================================================================
# 需要先放好 ckpt/data2vec/base_no_ft.pt（1.4GB）。
# conf/base.yaml:
#   input_format: repr
#   encoder: repcodec
#   quantizer: rvq
#   decoder: dac
#   vocoder: null
# conf/input/repr.yaml:    ssl_model_type: data2vec
# conf/inference.yaml:     resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/repcodec_rvq_data2vec --args.debug 1"

# ==============================================================================
# 8b) repcodec-rvq-hubert     repr + repcodec + rvq + dac   (SSL=hubert, 1024-dim)
# ==============================================================================
# 需要先放好 ckpt/hubert/hubert_large_ll60k.pt（3.8GB）。
# conf/base.yaml:
#   input_format: repr
#   encoder: repcodec
#   quantizer: rvq
#   decoder: dac
#   vocoder: null
# conf/input/repr.yaml:    ssl_model_type: hubert
# conf/inference.yaml:     resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/repcodec_rvq_hubert --args.debug 1"

# ==============================================================================
# 8c) repcodec-rvq-whisper    repr + repcodec + rvq + dac   (SSL=whisper, 1024-dim)
# ==============================================================================
# 需要先放好 ckpt/whisper/medium.pt（1.5GB）。Whisper encoder 输入要求 16kHz，
# 单条推理 audio 的 sample_rate 由 conf/inference.yaml 控制。
# conf/base.yaml:
#   input_format: repr
#   encoder: repcodec
#   quantizer: rvq
#   decoder: dac
#   vocoder: null
# conf/input/repr.yaml:    ssl_model_type: whisper
# conf/inference.yaml:     resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/repcodec_rvq_whisper --args.debug 1"

# ==============================================================================
# 9) mel-cosmos-rvq    melspec + cosmos + rvq + cosmos    (2D image-style mel codec)
# ==============================================================================
# cosmos 把 (n_mels, T) 当作 2D image 处理（Conv2d）。DynamicCodec.preprocess
# 会自动把 n_mels 从 100 pad 到 cosmos.yaml 里的 resolution=104，并在 decoder
# 输出后 crop 回 100，所以无需改 mel_model.n_mels。
# conf/base.yaml:
#   input_format: melspec
#   encoder: cosmos
#   quantizer: rvq
#   decoder: cosmos
#   vocoder: null     # 也可以用 vocos 让最终输出回到 wav 域
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/mel_cosmos_rvq --args.debug 1"


# ==============================================================================
# 训练任意组合（同样改完 conf/base.yaml 后切 state: train，再用 train.py）
# ==============================================================================
# conf/base.yaml:  state: train
# 训练前确认：
#   - data/manifests/vctk/{train,val,test}.csv 已经生成（scripts/build_vctk_manifest.py）
#   - conf/train.yaml 的 load.resume 与 ckpt 状态一致

# 单卡：
sudo docker exec $CTR bash -c \
  "cd /app && python train.py --conf_path conf/base.yaml --save_path runs/dac_rvq_train --args.debug 1"

# 多卡：
sudo docker exec $CTR bash -c \
  "cd /app && torchrun --nproc_per_node gpu train.py --conf_path conf/base.yaml --save_path runs/dac_rvq_train_ddp --args.debug 1"
