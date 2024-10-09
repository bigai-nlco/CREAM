factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5


prefix=/root/${task_name}/${model_type}_results

echo $scaling_type and $factor
python -u src/evaluation/eval_ppl.py \
    --model_name_or_path ${prefix}/4k-$((factor*4))k-${scaling_type}/checkpoint-${step} \
    --model_name 7b-4k-$((factor*4))k-${scaling_type} \
    --rope_scaling_type ${scaling_type} \
    --rope_scaling_factor ${factor} \
    --model_max_position_embeddings 4096 \
    --max_input_tokens 262144 \
    --min_input_tokens 262144 \
    --sliding_window_step 4096 \
    --window_length_list 65536 98304 131072 196608 262144 \
    --eval_nums 20 \
    --dataset_name pg19 \
    --path_to_dataset /root/public_datasets/pg19_long \
    --path_to_output_dir eval_output/PPL/${model_type}-${task_name}-${step}

python -u src/evaluation/eval_ppl.py \
    --model_name_or_path ${prefix}/4k-$((factor*4))k-${scaling_type}/checkpoint-${step} \
    --model_name 7b-4k-$((factor*4))k-${scaling_type} \
    --rope_scaling_type ${scaling_type} \
    --rope_scaling_factor ${factor} \
    --model_max_position_embeddings 4096 \
    --max_input_tokens 262144 \
    --min_input_tokens 262144 \
    --sliding_window_step 4096 \
    --window_length_list 65536 98304 131072 196608 262144 \
    --eval_nums 20 \
    --dataset_name books3 \
    --path_to_dataset /root/public_datasets/book3 \
    --path_to_output_dir eval_output/PPL/${model_type}-${task_name}-${step}
