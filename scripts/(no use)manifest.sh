#
# Generate manifest files for VCTK dataset
#
# 扫描指定目录下的音频文件
# 生成训练集和验证集的分割清单
# 记录每个音频文件的帧数信息
# 提供灵活的文件过滤和随机分割功能
#

DATASET=/mnt/speech/luyongkang/datasets/VCTK/sample/audio
# DATASET=./wav_file/input


DEST=./wav_file/manifest

python model/ssl/utils/wav2vec_manifest.py \
  $DATASET \
  --dest $DEST \
  --ext wav \
  --valid-percent 0