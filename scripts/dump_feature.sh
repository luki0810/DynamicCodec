# Example 2: dump from Whisper medium layer 18
export PYTHONPATH="$PWD:$PYTHONPATH"
layer=18
model_type=hubert
tsv_path=wav_file/manifest/train.tsv
ckpt_path=/mnt/speech/luyongkang/models/hubert/hubert_large_ll60k.pt
feat_dir=wav_file/ssl_repr

python3 model/ssl/utils/dump_feature.py \
    --model_type ${model_type} \
    --tsv_path ${tsv_path} \
    --ckpt_path ${ckpt_path} \
    --layer ${layer} \
    --feat_dir ${feat_dir} \
    --device cuda:0


# get 0_1.len and 0_1.npy in ${feat_dir}
# shape [T', D]
# D means SSL encoder's output (not SSL decoder's final output)
# in (ckpt_path=/mnt/speech/luyongkang/models/hubert/hubert_large_ll60k.pt) D = 1024