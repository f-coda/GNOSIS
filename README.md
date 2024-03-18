# GNOSIS   

GNOSIS is a learning approach that addresses the Minimum Vertex Cover problem through the combination of Graph Neural Networks and Deep Reinforcement Learning.

This solution combines the representation power of a Graph Neural Network approach with the ability of actor-critic Reinforcement Learning to provide strong solutions.

The code requires installing DGL ([Deep Graph Library](https://www.dgl.ai/)).   

  ## Network Τopologies & Metrics
  
  **Network topologies**:

 - Erdo-Renyi
 - Watts-Strogatz
 - Barabasi-Albert

**Evaluation metrics**:

- **Execution time**: refers to the total amount of time each algorithm requires to produce a solution
-  **Cost function**: This function calculates a cost based on the number of image replicas placed on the network as well as the transfer delays, in order to share the image between all network nodes
- **Vertex cover set size**: the size of vertices in vertex cover

## Configuration File

In `parameters_config.json` file, several variables can be configured:

    "episode": 5
    "network": "barabasi_albert",  
    "number_of_nodes": 64,  
    "probability": 0.5,  
    "degree": 1,  
    "knearest": 2  

`episode`: Episode is a sequence of interactions between an agent and the environment 

`network`: The name of the network (*choose between erdos_renyi, barabasi_albert or newman_watts_strogatz*)

`number_of_nodes`: The number of nodes of the graph

`probability`: The probability of adding a new edge for each edge (*only for Erdo-Renyi and Watts-Strogatz graphs*)

`degree`: Number of edges to attach from a new node to existing nodes (*only for Barabasi-Albert graphs*)

`knearest`: Each node is joined with its $k$ nearest neighbors in a ring topology (*only for Watts-Strogatz graphs*)

--
More information about setting variables on graphs can be found [here](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.newman_watts_strogatz_graph.html). 

## Usage  

```python3 find_MVC_drl.py --c parameters_config.json```  

## Cite Us

If you use the above code for your research, please cite our paper:

- [GNOSIS: Proactive Image Placement Using Graph Neural Networks & Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10255001?casa_token=oZPfryvDE1QAAAAA:G6QgrFGGSIt-JDOp0b6ZCs7MAQYNu_V5Kv99Q0yxSUzyXOTkV_x-11AE9J3Fg_qWT2bXVrQS)
       
      @inproceedings{theodoropoulos2023gnosis,
      title={GNOSIS: Proactive Image Placement Using Graph Neural Networks \& Deep Reinforcement     Learning},
      author={Theodoropoulos, Theodoros and Makris, Antonios and Psomakelis, Evangelos and Carlini,  Emanuele and Mordacchini, Matteo and Dazzi, Patrizio and Tserpes, Konstantinos},
      booktitle={2023 IEEE 16th International Conference on Cloud Computing (CLOUD)},
      pages={120--128},
      year={2023},
      organization={IEEE}
      } 
