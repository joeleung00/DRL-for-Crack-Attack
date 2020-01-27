from pylab import axes
import matplotlib.pyplot as plt
import statistics

f_origin = open("./full_dataorigin", "r")
f_addscore = open("./full_addscore", "r")
f_scorenet = open("./full_scorenet", "r")

origin = []
for line in f_origin:
    origin.append(float(line))
m = statistics.mean(origin)
sd = statistics.stdev(origin, xbar=m)
print(m)
print(sd)
addscore = []
for line in f_addscore:
    addscore.append(float(line))
m = statistics.mean(addscore)
sd = statistics.stdev(addscore, xbar=m)
print(m)
print(sd)
scorenet = []
for line in f_scorenet:
    scorenet.append(float(line))
m = statistics.mean(scorenet)
sd = statistics.stdev(scorenet, xbar=m)
print(m)
print(sd)

plt.boxplot([origin, addscore, scorenet])
ax = axes()
ax.set_xticklabels(["Original", "Rewarded", "Neural Network"])

plt.show()
