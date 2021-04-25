# NE_DeepWalk
DeeplWalk implementation with simple example.
implementation w/o ML framework such as Tensorflow or Pytorch.

![image](https://user-images.githubusercontent.com/35905280/96146155-f654ac80-0f40-11eb-8f95-1a6f334e2019.png)

## Requirments
networkx == 2.5

matplotlib

pandas


## Description

The files for embedding Karate network.
There is raw edgelist data for Karete.
the main script includes not only DeepWalk class but also data preprocessings.

Deep neural network embedding based on NLP Skip-gram model.
nodes are encoded with one-hot vector, than came through designed DNN.
the loss is designed to minimize the distance between **Random Walk Sequence**

For reducing complexity, Hierarchical softmax is adopted.
there are both version (w/o Hierarchical softmax , with H.S.).



