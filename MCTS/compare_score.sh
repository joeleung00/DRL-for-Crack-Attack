# /bin/bash
totalscore=0
for i in {1..100};
do
	score=$(python3 MCTS.py | tail -n 1)
	totalscore=$(echo $score + $totalscore | bc)
done
totalscore=$(echo $totalscore/100 | bc)
echo $totalscore
