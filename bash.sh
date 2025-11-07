export PYTHONPATH="$PWD:$PYTHONPATH"
python main.py \
--load_path conf/base.yaml \
--save_path runs/test \
--args.debug 1




export CUDA_VISIBLE_DEVICES=7
python train.py \
--load_path conf/base.yaml \
--save_path runs/test/ \
--args.debug 1




export CUDA_VISIBLE_DEVICES=0,7
torchrun \
--nproc_per_node gpu train.py \
--load_path conf/base.yaml \
--save_path runs/test/