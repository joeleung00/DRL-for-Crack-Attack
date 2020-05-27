#!/bin/bash
python td_learning.py \
--total_iterations 15 \
--step_per_iteration 50000 \
--initial_epsilon "0.9" \
--final_epsilon "0.1" \
--output_network "./network/duel_test.pth" \
--learning_rate "1e-3" \
--total_epoch 2 \
--gamma "0.6" \
--batch_size 256
#--input_network "./network/dueling_try3.pth"
