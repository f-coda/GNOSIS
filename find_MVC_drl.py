# -*- coding: utf-8 -*-
from discrete_actor_critic import DiscreteActorCritic
import torch
from MVC import MVC
from MVC_newman_watts_strogatz import MVC_NWS
from MVC_barabasi_albert import MVC_BA
import dgl
import torch.nn.functional as F
from Models import ACNet
import time
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import argparse
import sys
import maxmin
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--configFile", type=str, help="path and name to the configuration file")
args = vars(ap.parse_args())

params = args["configFile"]
params = json.loads(open(params).read())

if params["network"] == "erdos_renyi":
    mvc = MVC(params["number_of_nodes"], params["probability"])
if params["network"] == "barabasi_albert":
    mvc = MVC_BA(params["number_of_nodes"], params["degree"])
if params["network"] == "newman_watts_strogatz":
    mvc = MVC_NWS(params["number_of_nodes"], params["knearest"], params["probability"])

cuda_flag = False
alg = DiscreteActorCritic(mvc, cuda_flag)

num_episodes = params["episode"]

for i in range(num_episodes):
    T1 = time.time()
    log = alg.train()
    T2 = time.time()
    print('Epoch: {}. R: {}. TD error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'), 2),
                                                                np.round(log.get_current('TD_error'), 3),
                                                                np.round(log.get_current('entropy'), 3),
                                                                np.round(T2 - T1, 3)))
PATH = 'mvc_net.pt'
torch.save(alg.model.state_dict(), PATH)

random.seed(10)
np.random.seed(10)
sys.setrecursionlimit(100000)
imageSize = 3*1024*1024*1024

# 10737418240
bandwidthEthernet = 10*1024*1024*1024
# 26214400
bandwidthWifi = 25*1024*1024
# 524288
bandwidthlocalfile = 0.5*1024*1024

def getScore(graph,placement):
    score = len(placement) + sum([graph.edges[edge[0],edge[1]]['usage']/graph.edges[edge[0],edge[1]]['capacity'] for edge in graph.edges])
    return score

cuda_flag = False
if params["network"] == "erdos_renyi":
    mvc = MVC(params["number_of_nodes"], params["probability"])
if params["network"] == "barabasi_albert":
    mvc = MVC_BA(params["number_of_nodes"], params["degree"])
if params["network"] == "newman_watts_strogatz":
    mvc = MVC_NWS(params["number_of_nodes"], params["knearest"], params["probability"])

ndim = mvc.get_graph_dims()

G2 = mvc.return_graph()
print ("graphtest", G2)
NODES = mvc.N
nodes_activated = np.random.choice(NODES, NODES, replace=False)

edgeCapacities = {}
for edge in G2.edges:
    if edge[0] == edge[1]:
        edgeCapacities[edge] = bandwidthlocalfile
    elif random.random() < 0.7:
        edgeCapacities[edge] = bandwidthWifi
    else:
        edgeCapacities[edge] = bandwidthEthernet

# max-min fairness
output = maxmin.max_min_fairness(demands=list(edgeCapacities.values()), capacity=20000000000)
counter = 0
for key, value in edgeCapacities.items():
    edgeCapacities[key] = output[counter]
    counter = counter + 1

nx.set_edge_attributes(G2, values=edgeCapacities, name='capacity')
nx.set_edge_attributes(G2, values=0, name='usage')
nx.set_edge_attributes(G2, values=0, name='time')
nx.set_edge_attributes(G2, values=0, name='numImages')

start_time = time.time()

if cuda_flag:
    NN = ACNet(ndim,264,1).cuda()
else:
    NN = ACNet(ndim,264,1)
PATH = 'mvc_net.pt'
NN.load_state_dict(torch.load(PATH))

init_state,done = mvc.reset()
pos = nx.spring_layout(init_state.g.to_networkx(), iterations=20)

#### GCN Policy
state = dc(init_state)
if cuda_flag:
    state.g.ndata['x'] = state.g.ndata['x'].cuda()
sum_r = 0
T1 = time.time()
[idx1,idx2] = mvc.get_ilegal_actions(state)
while done == False:
    G = state.g
    [pi,val] = NN(G)
    pi = pi.squeeze()
    pi[idx1] = -float('Inf')
    pi = F.softmax(pi,dim=0)
    dist = torch.distributions.categorical.Categorical(pi)
    action = dist.sample()
    new_state, reward, done = mvc.step(state,action)
    [idx1,idx2] = mvc.get_ilegal_actions(new_state)
    state = new_state
    sum_r += reward

nodes_with_image = idx1.cpu().squeeze().numpy().tolist()
nearest_image = []
shortest_paths = nx.shortest_path(G2)

for active_node in nodes_activated:
    nearest_image.append(min(nodes_with_image, key=lambda x: len(shortest_paths[active_node][x])))
for i in range(len(nodes_activated)):
    sp = (shortest_paths[nodes_activated[i]][nearest_image[i]])
    # print(f"Shortest Path from {nodes_activated[i]} to {nearest_image[i]} is {sp}")
    for j in range(len(sp) - 1):
        G2[sp[j]][sp[j + 1]]['usage'] += imageSize
        G2[sp[j]][sp[j + 1]]['numImages'] = round(G2[sp[j]][sp[j + 1]]['usage'] / imageSize, 4)
        G2[sp[j]][sp[j + 1]]['time'] = G2[sp[j]][sp[j + 1]]['usage'] / G2[sp[j]][sp[j + 1]]['capacity']
        # print(f"Usage of channel {sp[j]} to {sp[j + 1]} is {G2[sp[j]][sp[j + 1]]['time'] * 100}")

print ("\n \t \t", "Statistics", "\n \t \t")
# Results
print("Execution Time: %s seconds" % (time.time() - start_time), "\n")
print(f"nodes nodes_with_image {nodes_with_image}", "\n")
print("Length of nodes with images", len(nodes_with_image), "\n")
print(f"Cost function value: {getScore(G2, nodes_with_image)}",  "\n")

# node_tag = state.g.ndata['x'][:,0].cpu().squeeze().numpy().tolist()
# nx.draw(state.g.to_networkx(), pos, node_color=node_tag, with_labels=True)
# plt.show()