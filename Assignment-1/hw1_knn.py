from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function
        self.lab=[]
        self.fea=[]

    def train(self, features: List[List[float]], labels: List[int]):
        for i in labels:
            self.lab.append(i)
        for j in features:
            self.fea.append(j)
            
            
            
    def predict(self, features: List[List[float]]) -> List[int]:
        
        import collections
        from collections import Counter
        
        v=[]
        for a,c in zip(features,self.lab):
            l1=[]
            l2=[]
            for g,h in zip(self.fea,self.lab):
                l1.append(g)
                l2.append(h)
            distances=[]
            for item,i in zip(l1,l2):
                dist=self.distance_function(a,item)
                distances.append([dist,i])
            votes=[i[1] for i in sorted(distances)[:self.k]]
            vote_result=Counter(votes).most_common(1)[0][0]
            v.append(vote_result)
        return v
         
        raise NotImplementedError

    
"""
    def predict(self, features: List[List[float]]) -> List[int]:
        
        import collections
        from collections import Counter
        
        v=[]
        for j,a,c in zip(range(len(self.lab)),features,self.lab):
            l1=[]
            l2=[]
            for g,h in zip(features,self.lab):
                l1.append(g)
                l2.append(h)
            del l1[j]
            del l2[j]
            distances=[]
            for item,i in zip(l1,l2):
                dist=self.distance_function(a,item)
                distances.append([dist,i])
            votes=[i[1] for i in sorted(distances)[:self.k]]
            vote_result=Counter(votes).most_common(1)[0][0]
            v.append(vote_result)
        return v
         
        raise NotImplementedError
"""

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
