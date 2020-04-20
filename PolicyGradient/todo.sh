#!/bin/bash
python pg.py \
--total_iterations 20 \
--step_per_iteration 500000 \
--observation_data 500000 \
--init_epsilon "0.8" \
--final_epsilon "0.1" \
--output_network "./network/try2.pth" \
--learning_rate "1e-3" \
--total_epoch 2 \
--batch_size 256 \
--NUM_OF_PROCESSES 10 \
--MAX_MEMORY_SIZE 1000000 \
--input_network "./network/test.pth"