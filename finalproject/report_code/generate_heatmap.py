#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np

# example usage:
#   ./generate_heatmap.py ./inputs/test01.txt

# infile should be the path to a file that has a list
# of x,y pairs, each separated by a newline. these are the files
# that contain training data which we were provided with
infile = sys.argv[1]

xy_pairs = []
with open(infile) as inf:
    for line in inf: xy_pairs.append([int(line.split(',')[0]), int(line.split(',')[1].strip())])

xs = [xy[0] for xy in xy_pairs]
ys = [xy[1] for xy in xy_pairs]

heatmap, _, _ = np.histogram2d(xs, ys, bins=50)

plt.clf()
plt.imshow(heatmap)
plt.colorbar()
plt.show()
