# /bin/bash
totalscore=0
for i in {1..500};
do
	score=$(python3 MCTS.py | tail -n 1) 
	totalscore=$(echo $score + $totalscore | bc)
done
totalscore=$(echo $totalscore/500 | bc)
echo $totalscore
