export HF_HOME=/path/to/huggingface
source /path/to/anaconda3/bin/activate /path/to/anaconda3/envs/dynamic
cd /path/to/DynamicCodec
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