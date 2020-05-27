#!/bin/bash
python sac.py \
--total_steps 1000001 \
--train_data_size 30000 \
--output_network "./network/test4/" \
--policy_lr "2e-4" \
--q_lr "2e-4" \
--total_epoch 1 \
--gamma "0.6" \
--batch_size 256 \
--alpha_lr "8e-4" \
--soft_tau "0.5" \
--start_learn 30000 \
--train_freq 20000 \
--MAX_REPLAY_MEMORY_SIZE 100000 \
--init_train_times "1" \
--init_log_alpha "-0.7"
#--input_network "../TD_learning/network/dueling_try3.pth"
#--update_target_freq 40000 \