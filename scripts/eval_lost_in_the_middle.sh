scaling_factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5


data_path_prefix=/root/public_datasets/lost_in_the_middle
model_path_prefix=/root/${task_name}/${model_type}_results


for gold_index in 0 18 37 54 74; do
    python -u src/evaluation/lost_in_the_middle/get_results.py \
        --input_path ${data_path_prefix}/kv-retrieval-75_keys.jsonl.gz \
        --model_name_or_path ${model_path_prefix}/4k-$((scaling_factor*4))k-${scaling_type}/checkpoint-${step} \
        --task_name kv \
        --gold_index ${gold_index} \
        --max_prompt_length $((scaling_factor*4096)) \
        --model_max_position_embeddings 4096 \
        --rope_scaling_factor ${scaling_factor} \
        --rope_scaling_type ${scaling_type} \
        --max_new_tokens 50 \
        --output_path eval_output/kv_predictions/${model_type}-${task_name}-${step}/kv_75_at_${gold_index}_4k_$((scaling_factor*4))k_${scaling_type}.txt
done


for gold_index in 0 34 69 104 139; do
    python -u src/evaluation/lost_in_the_middle/get_results.py \
        --input_path ${data_path_prefix}/kv-retrieval-140_keys.jsonl.gz \
        --model_name_or_path ${model_path_prefix}/4k-$((scaling_factor*4))k-${scaling_type}/checkpoint-${step} \
        --task_name kv \
        --gold_index ${gold_index} \
        --max_prompt_length $((scaling_factor*4096)) \
        --model_max_position_embeddings 4096 \
        --rope_scaling_factor ${scaling_factor} \
        --rope_scaling_type ${scaling_type} \
        --max_new_tokens 50 \
        --output_path eval_output/kv_predictions/${model_type}-${task_name}-${step}/kv_140_at_${gold_index}_4k_$((scaling_factor*4))k_${scaling_type}.txt
done



# folder_path=eval_output/kv_predictions/${model_type}-${task_name}-${step}

# file_list=$(ls "$folder_path"/${type}_75_*_"$scaling_type".txt)
# # file_list=$(ls "$folder_path"/${type}_140_*_"$scaling_type".txt)

# for file in $file_list; do
#     if [ -f "$file" ]; then
#         echo "Processing file: $file"

# 		python -u src/evaluation/longbench/eval.py --input-path $file

# 	fi
# done
