#! /bin/bash
for i in {1..2}; do
	python3 Sampling.py ${i} &
done
