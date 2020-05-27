#!/bin/bash
python sac.py \
--total_steps 2000000 \
--train_data_size 200000 \
--output_network "./network/try4/" \
--policy_lr "2e-4" \
--q_lr "2e-4" \
--total_epoch 1 \
--gamma "0.6" \
--batch_size 512 \
--alpha_lr "2e-3" \
--soft_tau "0.5" \
--start_learn 400000 \
--train_freq 100000 \
--MAX_REPLAY_MEMORY_SIZE 500000 \
--init_train_times "4" \
--init_log_alpha "-1"
#--input_network "../TD_learning/network/dueling_try3.pth"
#--update_target_freq 40000 \
#--input_network "./network/try3/" \