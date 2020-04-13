#!/bin/bash
python td_learning.py \
--total_iterations 50 \
--step_per_iteration 100000 \
--initial_epsilon "0.7" \
--final_epsilon "0.1" \
--output_network "./network/formal_size1.pth" \
--learning_rate "2e-3" \
--total_epoch 4 \
--gamma "0.5" \
--batch_size 128
#--input_network "./network/test2.pth" \