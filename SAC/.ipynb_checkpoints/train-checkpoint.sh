#!/bin/bash
python agent.py \
--output_network "./network/test.pth" \
--learning_rate "1e-3" \
--total_epoch 3 \
--gamma "0.6" \
--batch_size 256
#--input_network "./network/dueling_try1.pth"