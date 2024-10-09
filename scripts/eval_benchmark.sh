ZERO_SHOT="--tasks boolq,piqa,winogrande,truthfulqa_mc2"
HELLASWAG="--tasks hellaswag --num_fewshot 10"
ARC="--tasks arc_challenge --num_fewshot 25"

factor=$1
scaling_type=$2
model_type=$3
task_name=$4
step=$5

if [ $factor == "none" ]
then
    path_to_model=/scratch2/nlp/plm/Llama-2-7b-hf
    model_name=llama-2-7b-hf
else
    path_to_model=/root/${task_name}/${model_type}_results/4k-$((factor*4))k-$scaling_type/checkpoint-$step
    model_name=${task_name}-${model_type}-7b-4k-$((factor*4))k-${scaling_type}
fi

lm_eval \
    --model hf \
    --model_args "pretrained=${path_to_model},use_accelerate=True,dtype=bfloat16" \
    ${ZERO_SHOT} \
    --output_path eval_output/LM/${model_name}-zeroshot

lm_eval \
    --model hf \
    --model_args "pretrained=${path_to_model},use_accelerate=True,dtype=bfloat16" \
    ${HELLASWAG} \
    --output_path eval_output/LM/${model_name}-hellaswag

lm_eval \
    --model hf \
    --model_args "pretrained=${path_to_model},use_accelerate=True,dtype=bfloat16" \
    ${ARC} \
    --output_path eval_output/LM/${model_name}-ARC-C
