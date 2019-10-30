# /bin/bash
n=$1
if [ $# -ne 1 ]; then
	echo "Please enter the number of iteration"
else
	for i in $(seq 1 $n)
	do
		python3 MCTS.py 		
	done
fi
