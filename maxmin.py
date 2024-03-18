import networkx as nx
import numpy as np
import random
import numpy.random

random.seed(10)
numpy.random.seed(10)
imageSize = 3*1024*1024*1024

# # 10.737.418.240
# bandwidthEthernet = 10*1024*1024*1024
# # 26214400
# bandwidthWifi = 25*1024*1024
# # 524288
# bandwidthlocalfile = 0.5*1024*1024

# http://www.mathcs.emory.edu/~cheung/Courses/558/Syllabus/11-Fairness/Fair.html

def max_min_fairness(demands, capacity):
    capacity_remaining = capacity
    output = []

    for i, demand in enumerate(demands):
        share = capacity_remaining / (len(demands) - i)
        allocation = min(share, demand)

        if i == len(demands) - 1:
            allocation = max(share, capacity_remaining)

        output.append(allocation)
        capacity_remaining -= allocation

    return output
