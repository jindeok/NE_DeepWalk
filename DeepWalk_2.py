import numpy as np
import networkx as nx
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt

#data load, trimming
edge = pd.read_csv("karate_club.edgelist",sep = ' ', names = ['x','y','z'])
G2 = nx.read_gml('football.gml')
G = nx.Graph()
for i in range(edge.shape[0]):
    G.add_node(node_for_adding = edge['x'][i])
    G.add_node(node_for_adding = edge['y'][i])
    G.add_edge(edge['x'][i], edge['y'][i])
    
Karate = nx.to_numpy_matrix(G,nodelist = sorted(G.nodes()))
Karate = nx.to_numpy_matrix(G2)
OH = np.identity(len(Karate))

# class define
class DeepWalk:
    
    #initialization
    def __init__(self, data_set, w, d, r, t, lr):
        self.pi_i = np.random.random((len(data_set), d))
        self.pi_o = np.random.random((d, len(data_set)))
        self.h_vi = np.random.random((d, len(data_set) - 1))
        self.data_set = data_set # data_set matrix
        self.w = w # window size
        self.d = d # embedding size
        self.r = r # walks per vertex
        self.t = t # walk length
        self.LR = lr # learning rate
        self.Glen = len(data_set)
        
    
    def Cycle(self):
        
        for i in range(self.r):
            O = rnd.sample(list(range(len(Karate))), len(Karate)) 

            for vi in O:
                W_vi = self.RandomWalk(vi)
                self.SkipGram_SoftMax(W_vi)
           
        
    def RandomWalk(self, vi):        
        
        W_vi = np.zeros(self.t + 1, dtype = "i")
        W_vi[0] = vi #root of R.W
        
        for i in range(self.t):
            connected = np.where(Karate[W_vi[i]] == 1)
            connected = np.array(connected)[1]
            W_vi[i+1] = rnd.sample(list(connected), 1)[0]
           
            
        return W_vi
        
        
    def SkipGram_SoftMax(self, W_vi):
       

        for k in range(self.w, len(W_vi) - self.w ):
            
            EI = np.zeros((1,self.Glen))
            EH = np.zeros((1,2))          
            
            #feedforward
            ah = np.dot(OH[W_vi[k],:], self.pi_i)
            ah = np.expand_dims(ah, axis = 1)
            ao = np.dot(ah.T,self.pi_o)
            ay = softmax(ao)   
            
            #backprop
            for i in range(k - self.w, k + self.w + 1):
                 
                EI = EI + (ay - OH[W_vi[i],:])
            
            err1 = EI*ah             
            self.pi_o = self.pi_o - self.LR*err1      
            
            EH = np.dot(self.pi_o, EI.T)
            self.pi_i[W_vi[k],:] = self.pi_i[W_vi[k],:] - self.LR*EH.T
                
                

    

    

# activation func       
def softmax(arr):
    m = np.argmax(arr)
    arr = arr - m
    arr = np.exp(arr)
    return arr / np.sum(arr)

def sigmoid(arr):
    return 1.0/(1.0 + np.exp(-arr))


## main from now on ------------------------------------------
    
DW = DeepWalk(data_set = Karate, w = 3, d = 2, r = 10, t = 9, lr = 0.03)

for i in range(10):
    DW.Cycle()
#DW.Cycle()


# scattering plot

labels = []
labels_num = []
for i in range(len(Karate)):
    if(i in list([0,1,2,3,4,5,6,7,10,11,12,13,16,17,19,21])):
        labels.append("red")
        labels_num.append(0)
    else:
        labels.append("blue")
        labels_num.append(1)
		
df = pd.DataFrame(DW.pi_i,columns=["x","y"])
df_save = pd.DataFrame(DW.pi_i, columns = ["x1","x2"])

df['label'] = labels
df_save['labels_num'] = labels_num

fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.scatter(df['x'], df['y'], c = df['label'])

df_save.to_csv("embeddedKarate.csv")



