from matplotlib import pyplot as plt
import matplotlib

forums = []
with open('../model/selftrainscores.txt') as f:
	for line in f:
		if 'forums' in line:
			forums.append(float(line.split(')')[0].split(',')[0][1:]))
print forums
axes = plt.gca()


x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
plt.gca().yaxis.set_major_formatter(x_formatter)
axes.set_ylim([0.6374,0.6378])
plt.ylabel('Pearson Correlation')
plt.xlabel('Unlabelled data added')
plt.title('Self Training')
print forums[0:200]
print len(forums)
plt.plot(forums)
plt.show()


forums = []
with open('../model/tritrainscores.txt') as f:
	for line in f:
		if 'forums' in line:
			forums.append(float(line.split(')')[0].split(',')[0][1:]))
print forums
axes = plt.gca()


x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
plt.gca().yaxis.set_major_formatter(x_formatter)
plt.ylabel('Pearson Correlation')
plt.xlabel('Iterations')
plt.title('Tri-Training')

#axes.set_ylim([0.6374,0.6378])
print forums[0:200]
print len(forums)
plt.plot(forums)
plt.show()