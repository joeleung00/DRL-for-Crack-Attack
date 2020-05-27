#!/bin/bash
python td_learning.py \
--total_steps 1000000 \
--train_data_size 40000 \
--output_network "./network/test_per2.pth" \
--q_lr "5e-4" \
--total_epoch 2 \
--gamma "0.6" \
--batch_size 512 \
--soft_tau "0.5" \
--start_learn 40000 \
--train_freq 20000 \
--MAX_REPLAY_MEMORY_SIZE 1000000 \
--initial_epsilon 0.9 \
--final_epsilon 0.1 \
--initial_beta 0.001 \
--alpha 0.7