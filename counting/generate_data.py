# -*- coding: utf-8 -*-
import matplotlib 
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from pylab import rcParams
from matplotlib.patches import Rectangle
import random 
import os

FIG_NUMBER = 100
MAX_POINTS = 10
FIG_SIZE = 100
REC_HEI = 5
REC_WID = 5
my_dpi = 100

output = []
if not os.path.exists("../data/counting_samples"):
    os.makedirs("../data/counting_samples")

for num_fig in range(FIG_NUMBER):
	plt.figure(figsize=(FIG_SIZE/my_dpi, FIG_SIZE/my_dpi), dpi=my_dpi)
	currentAxis = plt.gca()
	# take random number of point between 
	npts = random.randint(0, MAX_POINTS);
	for i in range(npts):
		x = int(np.floor(FIG_SIZE * random.random()))
		y = int(np.floor(FIG_SIZE * random.random()))
		currentAxis.add_patch(Rectangle((x, y), REC_WID, REC_HEI, fill=True, alpha=1, color='#000000', edgecolor='#000000'))

	plt.xlim([0, FIG_SIZE])
	plt.ylim([0, FIG_SIZE])
	plt.axis('off')
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
	plt.savefig("../data/counting_samples/" + str(num_fig) + ".png")
	output += [[str(num_fig) + ".png", str(npts)]]


with open("../data/counting.csv", "w") as outfile:
	for row in output:
		outfile.write(",".join(row) + "\n")