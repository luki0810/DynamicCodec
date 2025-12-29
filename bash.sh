export HF_HOME=/data/L202500019/huggingface
source /data/L202500019/anaconda3/bin/activate /data/L202500019/anaconda3/envs/dynamic
cd /data/L202500019/DynamicCodec
export PYTHONPATH="$PWD:$PYTHONPATH"

# inference
python main.py \
--conf_path conf/base.yaml \
--save_path runs/inference \
--args.debug 1



python train.py \
--conf_path conf/base.yaml \
--save_path runs/dac-acp \
--args.debug 1



python train.py \
--conf_path conf/base.yaml \
--save_path runs/encodec+dac/ \
--args.debug 1



python train.py \
--conf_path conf/base.yaml \
--save_path runs/cosmos/ \
--args.debug 1



python train.py \
--conf_path conf/base.yaml \
--save_path runs/cosmos/ \
--args.debug 1



# multi-gpu (TODO)
torchrun \
--nproc_per_node gpu train.py \
--conf_path conf/base.yaml \
--save_path runs/test/