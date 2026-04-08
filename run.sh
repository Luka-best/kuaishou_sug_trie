# train
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py 1>train.log 2>err.log &

nohup torchrun --nproc_per_node=4 train.py ./outputs/qwen_nttp_logsumexp_sft_2epoch True 1>logs/train_logsumexp_2epoch.log 2>logs/train_logsumexp_2epoch.err &

nohup torchrun --nproc_per_node=4 train.py ./outputs/qwen_nttp_logsumexp_sft_prompt_v2 True 1>logs/train_logsumexp_prompt_v2.log 2>logs/train_logsumexp_prompt_v2.err &

nohup torchrun --nproc_per_node=4 train.py ./outputs/qwen_nttp_mean_sft True 1>logs/train_mean.log 2>logs/train_mean.err &

nohup torchrun --nproc_per_node=4 train.py ./outputs/qwen_wo_nttp_sft False 1>logs/train_wo_nttp.log 2>logs/train_wo_nttp.err &


# infer 
nohup python3 infer/batch_infer.py qwen_nttp_logsumexp_sft_prompt_v2/checkpoint-7813 greedy_infer_raw_logsumexp_prompt_v2 True 1> logs/batch_infer_greedy_raw_logsumexp_prompt_v2.log 2>logs/batch_infer_greedy_raw_logsumexp_prompt_v2.err &


CUDA_VISIBLE_DEVICES=1 nohup python3 infer/batch_infer.py qwen_wo_nttp_sft/checkpoint-7813 greedy_infer_wo_nttp False 1> logs/batch_infer_greedy_wo_nttp.log 2>logs/batch_infer_greedy_wo_nttp.err &

CUDA_VISIBLE_DEVICES=1 nohup python3 infer/batch_infer.py qwen_nttp_logsumexp_sft_2epoch/checkpoint-15626 greedy_infer_raw_logsumexp_2epoch True 1> logs/batch_infer_greedy_raw_logsumexp_2epoch.log 2>logs/batch_infer_greedy_raw_logsumexp_2epoch.err &


CUDA_VISIBLE_DEVICES=2 nohup python3 infer/wo_trie_batch_infer.py qwen_nttp_logsumexp_sft_2epoch/checkpoint-15626 wo_trie_greedy_infer_logsumexp_2epoch 1> logs/wo_trie_infer_greedy_logsumexp_2epoch.log 2>logs/wo_trie_infer_greedy_logsumexp_2epoch.err &


nohup python3 infer/beam_infer.py qwen_nttp_logsumexp_sft/checkpoint-7813 beam_infer_logsumexp 1> logs/beam_infer_logsumexp.log 2>logs/beam_infer_logsumexp.err &

CUDA_VISIBLE_DEVICES=3 nohup python3 infer/beam_cache_infer.py qwen_nttp_logsumexp_sft_prompt_v2/checkpoint-7813 beam_cache_infer_raw_logsumexp_prompt_v2 1> logs/beam_cache_infer_raw_logsumexp_prompt_v2.log 2>logs/beam_cache_infer_raw_logsumexp_prompt_v2.err &

CUDA_VISIBLE_DEVICES=3 nohup python3 infer/beam_cache_infer.py qwen_nttp_logsumexp_sft_prompt_v2/checkpoint-7813 beam_cache_infer_raw_logsumexp_prompt_v2_beam1 1> logs/beam_cache_infer_raw_logsumexp_prompt_v2_beam1.log 2>logs/beam_cache_infer_raw_logsumexp_prompt_v2_beam1.err &


# evaluate 
python3 evaluate.py outputs/wo_trie_greedy_infer_sum_results_checkpoint-7813.csv
python3 evaluate.py outputs/greedy_infer_sum_results_checkpoint-7813.csv

python3 evaluate.py outputs/wo_trie_greedy_infer_wo_nttp_results_checkpoint-7813.csv
python3 evaluate.py outputs/greedy_infer_wo_nttp_results_checkpoint-7813.csv

python3 evaluate.py outputs/wo_trie_greedy_infer_logsumexp_results_checkpoint-7813.csv
python3 evaluate.py outputs/greedy_infer_logsumexp_results_checkpoint-7813.csv

python3 evaluate.py outputs/beam_cache_infer_raw_logsumexp_results_checkpoint-7813.csv
python3 evaluate.py outputs/greedy_infer_raw_logsumexp_results_checkpoint-7813.csv

python3 evaluate.py outputs/greedy_infer_raw_logsumexp_2epoch_results_checkpoint-15626.csv
python3 evaluate.py outputs/beam_cache_infer_raw_logsumexp_2epoch_results_checkpoint-15626.csv

python3 evaluate.py outputs/beam_cache_infer_raw_logsumexp_prompt_v2_results_checkpoint-7813.csv
python3 evaluate.py outputs/greedy_infer_raw_logsumexp_prompt_v2_results_checkpoint-7813.csv
python3 evaluate.py outputs/wo_trie_greedy_infer_logsumexp_prompt_v2_results_checkpoint-7813.csv


#pciture
python3 loss_picture.py logs/train_wo_nttp.log
python3 loss_picture.py logs/train.log
python3 loss_picture.py logs/train_logsumexp.log
python3 loss_picture.py logs/train_logsumexp_2epoch.log
python3 loss_picture.py logs/train_logsumexp_2epoch.log
python3 loss_picture.py logs/train_logsumexp_prompt_v2.log
