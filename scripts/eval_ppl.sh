factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5


prefix=/root/${task_name}/${model_type}_results

if [ "$scaling_factor" == "none" ];
then
    echo Llama-2-7b-hf
    python -u src/evaluation/eval_ppl.py \
        --model_name_or_path /scratch2/nlp/plm/Llama-2-7b-hf \
        --model_name llama-2-7b-hf \
        --model_max_position_embeddings 4096 \
        --max_input_tokens 32768 \
        --min_input_tokens 32768 \
        --window_length_list 4096 8192 16384 32768 \
        --truncate \
        --dataset_name scrolls-gov_report \
        --path_to_dataset /root/public_datasets/gov_report \
        --path_to_output_dir eval_output/PPL/${model_type}-${task_name}

    python -u src/evaluation/eval_ppl.py \
        --model_name_or_path /scratch2/nlp/plm/Llama-2-7b-hf \
        --model_name llama-2-7b-hf \
        --model_max_position_embeddings 4096 \
        --max_input_tokens 32768 \
        --min_input_tokens 32768 \
        --window_length_list 4096 8192 16384 32768 \
        --truncate \
        --dataset_name proof-pile \
        --input_field input \
        --path_to_dataset /root/public_datasets/proof-pile \
        --path_to_output_dir eval_output/PPL/${model_type}-${task_name}

else
    echo $scaling_type and $factor
    python -u src/evaluation/eval_ppl.py \
        --model_name_or_path ${prefix}/4k-$((factor*4))k-${scaling_type}/checkpoint-${step} \
        --model_name 7b-4k-$((factor*4))k-${scaling_type} \
        --rope_scaling_type ${scaling_type} \
        --rope_scaling_factor ${factor} \
        --model_max_position_embeddings 4096 \
        --max_input_tokens 32768 \
        --min_input_tokens 32768 \
        --window_length_list 4096 8192 16384 32768 \
        --truncate \
        --dataset_name scrolls-gov_report \
        --path_to_dataset /root/public_datasets/gov_report \
        --path_to_output_dir eval_output/PPL/${model_type}-${task_name}-${step}

    python -u src/evaluation/eval_ppl.py \
        --model_name_or_path ${prefix}/4k-$((factor*4))k-${scaling_type}/checkpoint-${step} \
        --model_name 7b-4k-$((factor*4))k-${scaling_type} \
        --rope_scaling_type ${scaling_type} \
        --rope_scaling_factor ${factor} \
        --model_max_position_embeddings 4096 \
        --max_input_tokens 32768 \
        --min_input_tokens 32768 \
        --window_length_list 4096 8192 16384 32768 \
        --truncate \
        --dataset_name proof-pile \
        --input_field input \
        --path_to_dataset /root/public_datasets/proof-pile \
        --path_to_output_dir eval_output/PPL/${model_type}-${task_name}-${step}

fi
