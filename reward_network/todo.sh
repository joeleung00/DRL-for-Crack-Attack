#!/bin/bash
if [ $# -lt 1 ]
then
	echo "enter how many threads"
else
	n=$1
	for i in $(seq 1 1 ${n});do
		python action_reward_sampling.py --output_path "./output/data${i}" --num_sample "${2}" &
	done
fi
