bash scripts/run_CREAM.sh 8 linear llama2 5946 CREAM

bash scripts/run_CREAM_chat.sh 8 linear llama2_chat 5946 CREAM

bash scripts/eval_longchat_lines.sh 8 linear llama2 CREAM 1000

bash scripts/eval_lost_in_the_middle.sh 8 linear llama2 CREAM 1000

bash scripts/eval_needle.sh 8 linear llama2_chat CREAM 100

bash scripts/eval_longbench.sh 8 linear llama2_chat CREAM 100

bash scripts/eval_ppl.sh 8 linear llama2 CREAM 1000

bash scripts/eval_long_ppl.sh 64 linear llama2 CREAM 1000

bash scripts/eval_benchmark.sh 8 linear llama2 CREAM 1000
