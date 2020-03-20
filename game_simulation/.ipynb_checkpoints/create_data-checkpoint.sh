#! /bin/bash
if [ $# -lt 1 ]
then
	echo "enter your how may times"
else
	n=$1
	for i in $(seq 1  1 ${n}); do
		python3 Sampling.py ${i} &
	done
fi
