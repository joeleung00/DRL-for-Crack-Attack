#!/bin/bash
python td_learning.py \
--total_steps 1000001 \
--train_data_size 30000 \
--output_network "./network/small_board_td.pth" \
--q_lr "2e-4" \
--total_epoch 1 \
--gamma "0.6" \
--batch_size 512 \
--soft_tau "0.5" \
--start_learn 30000 \
--train_freq 20000 \
--MAX_REPLAY_MEMORY_SIZE 100000 \
--initial_epsilon 0.9 \
--final_epsilon 0.1