# -*- coding: utf-8 -*-
import matplotlib 
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from pylab import rcParams
from matplotlib.patches import Rectangle
import random 
import os
import glob


FIG_NUMBER = 100
MAX_POINTS = 10
FIG_SIZE = 10
REC_HEI = 1
REC_WID = 1

output = []
if not os.path.exists("../data/counting_samples"):
    os.makedirs("../data/counting_samples")

for num_fig in range(FIG_NUMBER):
	fig = plt.figure(frameon=False)
	fig.set_size_inches(FIG_SIZE,FIG_SIZE)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	# take random number of point between 
	npts = random.randint(0, MAX_POINTS);
	for i in range(npts):
		x = int(np.floor(FIG_SIZE * random.random()))
		y = int(np.floor(FIG_SIZE * random.random()))
		ax.add_patch(Rectangle((x, y), REC_WID, REC_HEI, fill=True, alpha=1, color='#000000', edgecolor='#000000'))

	plt.xlim([0, FIG_SIZE])
	plt.ylim([0, FIG_SIZE])
	plt.axis('off')
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
	plt.savefig("../data/counting_samples/" + str(num_fig) + ".png", dpi=1)
	output += [[str(num_fig) + ".png", str(npts)]]


with open("../data/counting.csv", "w") as outfile:
	outfile.write("file,number\n")
	for row in output:
		outfile.write(",".join(row) + "\n")
