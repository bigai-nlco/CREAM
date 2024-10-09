factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5


prefix=/root/${task_name}/${model_type}_results

python -u src/evaluation/needle/eval.py \
    --data_root /root/public_datasets/needle \
    --output_dir eval_output/needle/${model_type}-${task_name}-${step} \
    --model_name_or_path ${prefix}/4k-$((factor*4))k-${scaling_type}/checkpoint-${step} \
    --model_max_position_embeddings 4096 \
    --rope_scaling_type ${scaling_type} \
    --rope_scaling_factor ${factor} \
    --min_length 1024 \
    --max_length 32700
