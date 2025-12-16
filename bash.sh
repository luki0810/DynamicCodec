export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME=/data/L202500019/huggingface
source /data/L202500019/anaconda3/bin/activate /data/L202500019/anaconda3/envs/dynamic

# inference
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py \
--conf_path conf/base.yaml \
--save_path runs/inference \
--args.debug 1


export CUDA_VISIBLE_DEVICES=1
python train.py \
--conf_path conf/base.yaml \
--save_path runs/dac+dac \
--args.debug 1


export CUDA_VISIBLE_DEVICES=7
python train.py \
--conf_path conf/base.yaml \
--save_path runs/encodec+dac/ \
--args.debug 1



export CUDA_VISIBLE_DEVICES=7
python train.py \
--conf_path conf/base.yaml \
--save_path runs/test/ \
--args.debug 1

python train.py \
--conf_path conf/base.yaml \
--save_path runs/mel/ \
--args.debug 1




export CUDA_VISIBLE_DEVICES=0,7
torchrun \
--nproc_per_node gpu train.py \
--conf_path conf/base.yaml \
--save_path runs/test/