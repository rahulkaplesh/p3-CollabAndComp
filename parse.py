import matplotlib.pyplot as plt
import numpy as np

fileHandle = open('Log_File.txt', 'r')
avgScore = []
for line in fileHandle.readlines() :
    avgScore.append(float(line.split(' ')[-1].strip('\n')))

print(avgScore)
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(np.arange(1, len(avgScore)+1), avgScore, label='score')
plt.ylabel('Average Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left')
plt.savefig('plot.png')
