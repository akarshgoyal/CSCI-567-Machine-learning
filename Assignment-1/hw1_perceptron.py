from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration
        self.p=[]
        self.eq=[]

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        l=0
        m = len(features)
        g=self.margin/2
        h=(self.margin/2)*-1
        epi=0.000001
        a = np.matrix(features)
        z=np.ones((m,1))
        #c=np.concatenate((z,a),axis=1)
        q=a.tolist()
        for dy in q[0]:
            l=l+1
            
        self.p=np.zeros((l,1))
        self.p=self.p.transpose()
        e=np.array(labels)
        for _ in range(self.max_iteration):
            d=0
            while (d<m):
                t=np.dot(a[d],self.p.transpose())
                if(t>0):
                    y=1
                else:
                    y=-1
                v=np.linalg.norm(self.p)+epi
                s=np.linalg.norm(a[d])
                if((h<(t/v))and(g>(t/v))):
                    
                    self.p=self.p+(np.dot(e[d],a[d]))/s
                    d=d+1
                elif(np.dot(self.p,(np.dot(a[d],e[d])).transpose())<0):
                    self.p=self.p+(np.dot(e[d],a[d]))/s
                    d=d+1
                else:
                    d=d+1
          
        if((h<(t/v))and(g>(t/v))):
            return 0
        elif(np.dot(self.p,(np.dot(a[d-1],e[d-1])).transpose())<0):
            return 0
        else:
            return 1
                
        
        raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        m = len(features)
        a=np.matrix(features)
        z=np.ones((m,1))
        #c=np.concatenate((z,a),axis=1)
        x=np.dot(a,self.p.transpose())
        y=x.tolist()
        e=[]
        f=[]
        for item in y:
            for gh in item:
                e.append(gh)
        for sh in e:
            if (sh>0):
                f.append(1)
            else:
                f.append(-1)
                
        return f
        
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        
        eg=self.p.tolist()
        for item in eg:
            for i in item:
                self.eq.append(i)
        return self.eq
    