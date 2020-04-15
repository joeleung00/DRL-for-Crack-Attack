#!/bin/bash
python td_learning.py \
--total_iterations 10 \
--step_per_iteration 100000 \
--initial_epsilon "0.3" \
--final_epsilon "0.15" \
--output_network "./network/double_q_try3.pth" \
--learning_rate "1e-3" \
--total_epoch 2 \
--gamma "0.6" \
--batch_size 256 \
--input_network "./network/double_q_try3.pth" \