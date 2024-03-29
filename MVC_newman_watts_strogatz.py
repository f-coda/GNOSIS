import dgl
import torch
import networkx as nx
import numpy as np


class State:
    def __init__(self, g, visited):
        self.N = g.number_of_nodes()
        self.g = g
        self.visited = visited


def init_state(N, K, P):
    # erdos_renyi_graph
    # g = nx.fast_gnp_random_graph(N,P)

    # barabasi_albert_graph
    g = nx.newman_watts_strogatz_graph(N, K, P)

    g = dgl.DGLGraph(g)
    norm_card = torch.Tensor(np.array(g.in_degrees() + g.out_degrees()) / g.number_of_nodes()).unsqueeze(-1)
    g.ndata['x'] = torch.cat((torch.zeros((N, 1)), norm_card), dim=1)
    visited = torch.zeros((1, N)).squeeze()
    return g, visited


def init_state_to_return_graph(N, K, P):
    # erdos_renyi_graph
    # g = nx.fast_gnp_random_graph(N,P)

    # barabasi_albert_graph
    g = nx.newman_watts_strogatz_graph(N, K, P)

    return g


class MVC_NWS:
    def __init__(self, N, K, P):
        self.N = N
        self.K = K
        self.P = P
        [g, visited] = init_state(self.N, self.K, self.P)
        G2 = init_state_to_return_graph(self.N, self.K, self.P)
        self.init_state = State(g, visited)

    def return_graph(self):
        G2 = init_state_to_return_graph(self.N, self.K, self.P)
        return G2

    def get_graph_dims(self):
        return 2

    def reset(self):
        [g, visited] = init_state(self.N, self.K,self.P)
        state = State(g, visited)
        done = False
        return state, done

    def reset_fixed(self):
        done = False
        return self.init_state, done

    def get_ilegal_actions(self, state):
        idx1 = (state.visited == 1.).nonzero()
        idx2 = (state.visited == 0.).nonzero()
        return idx1, idx2

    def step(self, state, action):
        done = False
        state.g.ndata['x'][action, 0] = 1.0 - state.g.ndata['x'][action, 0]
        state.visited[action.item()] = 1.0
        reward = torch.Tensor(np.array([-1])).squeeze()

        edge_visited = \
        torch.cat((state.visited[state.g.edges()[0]].unsqueeze(-1), state.visited[state.g.edges()[1]].unsqueeze(-1)),
                  dim=1).max(dim=1)[0]
        if edge_visited.mean().item() == 1.0:
            done = True
        return state, reward, done