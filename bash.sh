#!/usr/bin/env bash
# DynamicCodec — 8 component combinations, run reference.
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
# 8) repcodec-rvq      repr + repcodec + rvq + dac    (SSL features → codec)
# ==============================================================================
# 需要先把 SSL ckpt 放到 ckpt/data2vec/base_no_ft.pt（对应 conf/input/ssl_model/data2vec.yaml）
# conf/base.yaml:
#   input_format: repr
#   encoder: repcodec
#   quantizer: rvq
#   decoder: dac
#   vocoder: null
# conf/inference.yaml: resume: false
sudo docker exec $CTR bash -c \
  "cd /app && python main.py --conf_path conf/base.yaml --save_path runs/repcodec_rvq --args.debug 1"


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
