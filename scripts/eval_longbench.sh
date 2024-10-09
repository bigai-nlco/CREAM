scaling_factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5


prefix=/root/${task_name}/${model_type}_results

if [ "$scaling_factor" == "none" ]
then
    echo Llama-2-7b-chat-hf
    python -u src/evaluation/longbench/get_results.py \
        --model_name_or_path /scratch2/nlp/plm/Llama-2-7b-chat-hf \
        --model_max_position_embeddings 4096 \
        --data_root_path /root/public_datasets/LongBench \
        --out_path_root eval_output/LongBench/${model_type}-${task_name}

else
    echo $scaling_type and $scaling_factor
    python -u src/evaluation/longbench/get_results.py \
        --model_name_or_path ${prefix}/4k-$((scaling_factor*4))k-${scaling_type}/checkpoint-${step} \
        --rope_scaling_type ${scaling_type} \
        --rope_scaling_factor ${scaling_factor} \
        --model_max_position_embeddings 4096 \
        --data_root_path /root/public_datasets/LongBench \
        --out_path_root eval_output/LongBench/${model_type}-${task_name}-${step}

fi


# folder_path=eval_output/LongBench/${model_type}-${task_name}-${step}

# python -u src/evaluation/longbench/eval.py \
#     --model_name ${model_type}-${task_name}-${scaling_type} \
#     --input_path ${folder_path} \
#     --scaling_type ${scaling_type}
