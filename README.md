# NE_DeepWalk
First network embedding implementation via DNN

![image](https://user-images.githubusercontent.com/35905280/96146155-f654ac80-0f40-11eb-8f95-1a6f334e2019.png)

## Requirments
networkx == 2.5
matplotlib
pandas


## Description

Deep neural network embedding based on NLP Skip-gram model.
nodes are encoded with one-hot vector, than came through designed DNN.

For reducing complexity, Hierarchical softmax is adopted.
there are both version (w/o Hierarchical softmax , with H.S.).

As for H.S. there are also 2 version based on data structure
1. complete tree
2. Huffman tree

Huffman tree is more effective on memory consumption, please refer to this.
