factor=$1
rope_type=$2
model_type=$3
port=$4
task_name=$5


data_path_prefix=/root/public_datasets
 
deepspeed \
    --include localhost:0,1 --master_port $port src/train.py \
    --model_name_or_path /scratch2/nlp/plm/Llama-2-7b-hf \
    --train_data_path $data_path_prefix/pile_4k_train \
    --valid_data_path $data_path_prefix/pile_val \
    --output_dir /root/${task_name}/${model_type}_results/4k-$((factor*4))k-${rope_type} \
    --max_steps 1000 \
    --model_max_position_embeddings 4096 \
    --rope_scaling_type ${rope_type} \
    --rope_scaling_factor $factor \
    --inference_length 16384 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train True \
    --do_eval True \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 500 \
    --load_best_model_at_end True \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed src/train_configs/deepspeed_config.json \
    --task_name ${task_name}
