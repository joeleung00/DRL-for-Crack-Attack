from pylab import axes
import matplotlib.pyplot as plt
import statistics

f_random = open("./random1", "r")
f_policy = open("./policy1", "r")

random = []
for line in f_random:
    random.append(float(line))
m = statistics.mean(random)
sd = statistics.stdev(random, xbar=m)
print(m)
print(sd)
policy = []
for line in f_policy:
    policy.append(float(line))
m = statistics.mean(policy)
sd = statistics.stdev(policy, xbar=m)
print(m)
print(sd)

plt.boxplot([random, policy])
ax = axes()
ax.set_xticklabels(["random", "policy"])

plt.show()
