#!/bin/bash
python sac.py \
--total_iterations 15 \
--step_per_iteration 25000 \
--initial_epsilon "0.9" \
--final_epsilon "0.1" \
--output_network "./network/test.pth" \
--learning_rate "1e-3" \
--total_epoch 4 \
--gamma "0.6" \
--batch_size 256 \
--NUM_OF_PROCESSES 2
#--input_network "../TD_learning/network/dueling_try3.pth"