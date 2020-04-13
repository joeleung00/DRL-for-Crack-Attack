python agent.py \
--total_epoch 4 \
--output_network "./network/train4.pth" \
--input_network "./network/train3.pth" \
--learning_rate "5e-4" \
--batch_size 128

##train1,2 gamma = 0
##train3 gamma = 0.6, c = 0.1
## train4 gamma =0.6, c = 0.1, is using 20Mdata