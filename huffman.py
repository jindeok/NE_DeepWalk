
import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
import networkx as nx
import neptune




class Node:
    def __init__(self, node, freq):
        self.node = node
        self.freq = freq
        self.left = None
        self.right = None
        self.t = 0
        self.code = []
        self.vec = np.random.uniform(0, 1, (2, 1))
    def __lt__(self, other):
        if(other == None):
            return -1
        if(not isinstance(other, Node)):
            return -1
        return self.freq > other.freq

class DeepWalk():
    # Constructor
    def __init__(self, dimension, learningRate, walkLength, 
                 windowSize, walksPerVertex, A, node_num, root, path):
        self.dimension = dimension
        self.learningRate = learningRate
        self.t = walkLength
        self.w = windowSize
        self.gamma = walksPerVertex
        self.fi = np.random.random((node_num, dimension))
        self.A = A
        self.node_num = node_num   
        self.loss = 0
        self.lossArr = []
        self.root = root
        self.path = path

    def embedding(self, epoch_num):
        for num in range(epoch_num):
            if num % 2 == 0 :
                print("epoch:", num)
            # add loss to loss array
            for kk in range(self.gamma):
                O = np.arange(0, self.node_num, 1)
                np.random.shuffle(O)
                for element in O:
                    W = self.RandomWalk(element)
                    self.SkipGram(W)
            self.lossArr.append(self.loss)
            # renew loss
            self.loss = 0  
                
    def RandomWalk(self, vi):
        tnum = 1
        w_array = [(vi)]
        #w_array.append(vi)
        vertex = vi
        randlist = np.arange(0, self.node_num, 1)
        while(tnum < self.t):
            np.random.shuffle(randlist)
            for i in randlist:
                if (A[vertex][i] == 1):
                    w_array.append(i)
                    vertex = i
                    tnum += 1
                    break
        return w_array
    
    def makeWindow(self, cur, W):
        uk = []
        if (cur < 3):
            uk = W[: cur + self.w]
        elif(2 < cur < 8):
            uk = W[cur - self.w : cur + self.w]
        else:
            uk = W[cur - self.w :]
        return uk
    
    def SkipGram(self, W):
        cur = 0
        for vj in W:
            # make -w to w
            uk = self.makeWindow(cur, W)
            cur += 1
    
            # make one-hot vector of target word
            target = np.zeros([self.node_num, 1]) # 34x1
            target[vj] = 1
            
            # define h, u, y
            h = np.dot(self.fi.T, target) # 2x1

            for n in uk: 
                if (vj == n):
                    continue
                EH = np.zeros([2, 1]) # 2x1
                curNode = self.root
                for i in self.path[n]:
                    if (i == '1'):
                        t = 1
                    else:
                        t = 0
                    sig = self.sigmoid(np.dot(curNode.vec.T, h)) - t
                    curNode.vec -= learningRate * np.dot(h, sig)
                    EH += np.dot(curNode.vec, sig)
                    if (i == '1'):
                        curNode = curNode.left
                    else:
                        curNode = curNode.right     
                self.fi[vj, :] = self.fi[vj, :] - (self.learningRate * EH.T)
    
    def sigmoid (self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__" :
    
    def get_frequency_dict(A):
        frequency = {}
        for i in range(len(A)):
            frequency[i] = int(np.sum(A[i,:]))
        return frequency
      
    def make_heap(heap, frequency):
    	for key in frequency:
    		node = Node(key, -frequency[key])
    		heapq.heappush(heap, node)
           
    def merge_nodes(heap):
        while(len(heap) > 1):
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            node1.t = 1
            node2.t = 0
            heapq.heappush(heap, merged)
    
    def nodeindex(node, s):
        if ((node.left == None) and (node.right == None) and isinstance(node.node, int)):
            #print(node.node, s)
            path[node.node] = s
        if (node.left != None):
            nodeindex(node.left, s + "1")
        if (node.right != None):
            nodeindex(node.right, s + "0")    
    
    for i in range(20):
        
        G = nx.grid_2d_graph(2,2*i)
        A = nx.to_numpy_array(G)  
    
        epoch_num = 1
        dimension = 2
        learningRate = 0.02
        t = 10 # walkLength
        w = 3 # windowSize
        gamma = 5 # walksPerVertex
        node_num = len(A)
        path = [None] * node_num
        heap = []

        
        frequency = get_frequency_dict(A)
        make_heap(heap, frequency)
        merge_nodes(heap)
        nodeindex(heap[0], "")
        start_time = time.time()
        rw = DeepWalk(dimension, learningRate, t, w, gamma, A, node_num, heap[0], path)
        rw.embedding(epoch_num)
        print("time:", time.time() - start_time)
        neptune.log_metric('Nodes:',len(A))
        neptune.log_metric('Huffman time:', time.time() - start_time)


    # Divide the nodes into 2 groups
    # Group 1
#    x1 = []
#    y1 = []
#    # Group 2
#    x2 = []
#    y2 = []
#    group1 = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
#    group2 = [8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
#    for i in range(node_num):
#        if i in group1:
#            x1.append(rw.fi[i,0])
#            y1.append(rw.fi[i,1])
#        else:
#            x2.append(rw.fi[i,0])
#            y2.append(rw.fi[i,1])
#    # plot Group 1
#    plt.scatter(x1, y1, color = "r", label = "Group 1")
#    for i, txt in enumerate(group1):
#        plt.annotate(txt, (x1[i], y1[i]))
#    # plot Group 2
#    plt.scatter(x2, y2, color = "b", label = "Group 2")
#    for i, txt in enumerate(group2):
#        plt.annotate(txt, (x2[i], y2[i]))
#    plt.title('Nodes in Karate Network')
#    plt.legend()
#    plt.show()
    

