
factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5


prefix=/root/${task_name}/${model_type}_results

if [ "$scaling_factor" == "none" ]
then
    python -u src/evaluation/longchat_lines/eval.py \
        --model_name_or_path /scratch2/nlp/plm/Llama-2-7b-hf \
        --model_name llama-2-7b-hf \
        --model_max_position_embeddings 4096 \
        --output_dir eval_output/longchat-lines/${model_type}-${task_name}

else
    echo $scaling_type and $factor
    python -u src/evaluation/longchat_lines/eval.py \
        --model_name_or_path ${prefix}/4k-$((factor*4))k-${scaling_type}/checkpoint-${step} \
        --model_name 7b-4k-$((factor*4))k-${scaling_type} \
        --rope_scaling_type ${scaling_type} \
        --rope_scaling_factor ${factor} \
        --model_max_position_embeddings 4096 \
        --output_dir eval_output/longchat-lines/${model_type}-${task_name}-${step}

fi
