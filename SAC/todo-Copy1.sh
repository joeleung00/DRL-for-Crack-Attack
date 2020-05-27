#!/bin/bash
python sac.py \
--total_iterations 20 \
--step_per_iteration 500000 \
--output_network "./network/try2/" \
--input_network "./network/try1/" \
--policy_lr "1e-4" \
--q_lr "1e-4" \
--total_epoch 4 \
--gamma "0.6" \
--batch_size 256 \
--alpha "0.5" \
--soft_tau "0.5"
#--load_observe_data "./input/try1"
#--observation_data 0 \
#--save_observation "./input/try1"
#--input_network "../TD_learning/network/dueling_try3.pth"
#--load_observe_data "./input/test" \