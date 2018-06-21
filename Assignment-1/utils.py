from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    t=[]
    for item in y_pred:
        for i in item:
            t.append(i)
    e=len(y_true)
    d=0
    for f, b in zip(y_true, t):
        a=f-b
        c=a**2
        d=d+c
        
    g=d/e
    return g

    raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    #print(len(real_labels))
    #print(len(predicted_labels))
    #assert len(real_labels) == len(predicted_labels)
    
    outputr = []
    outputp = []
    for x in real_labels:
        if x not in outputr:
            outputr.append(x)
        
    a=outputr[0]
    b=outputr[1]
    tp=0;
    fn=0;
    fp=0;
    
    if(a==1):
        for c,d in zip(real_labels, predicted_labels):
            if((c==a)and(d==a)):
                tp=tp+1
                
            elif((c==a)and(d==b)):
                fn=fn+1
                
            elif((c==b)and(d==a)):
                fp=fp+1
                
    elif(b==1):
        for e,f in zip(real_labels, predicted_labels):
            if((e==b)and(f==b)):
                tp=tp+1
                
            elif((e==b)and(f==a)):
                fn=fn+1
                
            elif((e==a)and(f==b)):
                fp=fp+1
    if((tp==0)and(fp==0)):
        precision=0
    else:
        precision=tp/(tp+fp)
    
    recall=tp/(tp+fn)
    
    
    if((recall==0)and(precision==0)):
        h=0
    else:
        h=(2*recall*precision)/(recall+precision)
    
    return h
               
    raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    
    a=[]
    
    for c in features:
        b=[]
        for i in range(1,k+1):
            for d in c:
                e=d**i
                b.append(e)
            
        a.append(b)
    #print(a)            
    return a
    
    raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    d=0
    for f, b in zip(point1, point2):
        c=(f-b)**2
        d=d+c
        
    e=d**(0.5)
    return e
    raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    d=0
    for f, b in zip(point1, point2):
        c=f*b
        d=d+c
    
    return d
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    import math
    d=0
    for f, b in zip(point1, point2):
        c=(f-b)**2
        d=d+c
        
    e=d
    a=e*(-0.5)
    g=math.exp(a)
    h=g*-1
    
    return h
    raise NotImplementedError
    
    



class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        
        #f=0
        a=[]
        #b=[]
        for c in features:
            f=0
            b=[]
            for d in c:
                e=d**2
                f=f+e
            g=f**(0.5)
            for h in c:
                j=h/g
                b.append(j)
            a.append(b)
        
        return a
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.count=0
        self.maxi=[]
        self.mini=[]
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        
        j=[]
        #l=[]
        a=len(features[0])
        print(a)
        b=len(features)
        data=np.array(features).T
        if self.count==0:
            for i in range(len(data)):
                self.maxi.append(np.max(data[i]))
                self.mini.append(np.min(data[i]))
            self.count=self.count+1
        #print(len(self.maxi))
        for i in range(b):
            l=[]
            for k in range(a):
                c=(features[i][k]-self.mini[k])/(self.maxi[k]-self.mini[k])
                l.append(c)
            j.append(l)
        t=np.matrix(j)
        #h=t.transpose()
        f=t.tolist()
        return f
        
        raise NotImplementedError