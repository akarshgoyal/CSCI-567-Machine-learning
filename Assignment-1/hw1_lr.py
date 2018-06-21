from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.e=[]
        self.f=[]

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        b=len(features)
        a=numpy.matrix(features)
        z=numpy.ones((b,1))
        c=numpy.concatenate((z,a),axis=1)
        d=numpy.matrix(values)
        w=numpy.dot(numpy.linalg.pinv(c),d.transpose())
        v=w.tolist()
        for item in v:
            self.e.append(item)


    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        g=self.e
        b=len(features)
        q=[]
        a=numpy.matrix(features)
        z=numpy.ones((b,1))
        c=numpy.concatenate((z,a),axis=1)
        h=numpy.matrix(g)
        x=numpy.dot(c,h)
        y=x.tolist()
        return y
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        for item in self.e:
            self.f.append(item)
        
        return self.f
        raise NotImplementedError


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.e=[]
        self.f=[]

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        b=len(features)
        a=numpy.matrix(features)
        z=numpy.ones((b,1))
        c=numpy.concatenate((z,a),axis=1)
        g=len(c[0])
        d=numpy.matrix(values)
        p=numpy.linalg.pinv((numpy.dot(c.transpose(),c))+(numpy.dot(self.alpha,numpy.identity(g))))
        w=numpy.dot(p,(numpy.dot(c.transpose(),d.transpose())))
        v=w.tolist()
        for item in v:
            self.e.append(item)
        
        

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        g=self.e
        b=len(features)
        a=numpy.matrix(features)
        z=numpy.ones((b,1))
        c=numpy.concatenate((z,a),axis=1)
        h=numpy.matrix(g)
        x=numpy.dot(c,h)
        y=x.tolist()
        return y
        raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        for item in self.e:
            self.f.append(item)
        
        return self.f
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
