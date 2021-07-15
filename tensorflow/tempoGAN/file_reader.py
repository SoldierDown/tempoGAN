from pathlib import Path
from sys import flags
import math
import matplotlib.pyplot as plt
import os

image_base_path = str(Path().absolute())
dir_base_path = str(image_base_path) + '/../2ddata_gan'
cost_file_path = dir_base_path + '/test_80_results.txt'
if os.path.exists(cost_file_path):
    print('yo')

exit()


# save directories in a list
models = []
costs = []

cost_file = open(cost_file_path,'r')
cost_lines = cost_file.read().splitlines()

for cost_line in cost_lines:
    vals = cost_line.split(' ')
    print('{} {}'.format(vals[0], vals[1]))
    model, cost = int(vals[0]), float(vals[1])
    models.append(model)
    costs.append(cost)    

plt.plot(models,costs)
plt.xlabel("model number")
plt.ylabel("cost")
plt.savefig(dir_base_path + '/test_80_costs.png')
