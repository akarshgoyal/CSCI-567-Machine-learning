import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
        # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        ########################################################
        # TODO: implement "predict"
        ########################################################
        be=self.betas
        be=np.array(be)
        he=[]
        for i in range(len(self.clfs_picked)):
                j=self.clfs_picked[i].predict(features)
                he.append(j)

        he=np.array(he)
        c=np.dot(be,he)
        d=np.sign(c)
        g=d.tolist()
        return g

class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
        
    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # TODO: implement "train"
        ############################################################
        #b=[]
        #for _ in range(len(labels)):
            #b.append(1/len(labels))

        b=np.ones(len(features))/len(features)  # (n, )
        c=np.array(b).reshape(-1, 1)
        #lab=np.array(labels).reshape(-1,1)
        lab=np.array(labels)
        h=[]
        for i in range(len(self.clfs)):
            j=(list(self.clfs))[i].predict(features)
            h.append(j)
        g=np.array(h)
        
        
        lae=np.array(labels)
        
        ge=[]
        """
        for y_pred in g:
            go=(y_pred==lae).astype(int)
            #print(go)
            gt=go.tolist()
            for item in gt:
                ge.append(item)
                """
        for x in g:
            ge.append([int(i) for i in (x!=lab)])

        ge=np.array(ge)
        
        #print(ge)
        #print(ge.shape)
        #count=0
        for _ in range(self.T):
            #print(ge.shape)
            s=np.dot(ge,c)
            #count=count+1
            #print(count)
            #print(s)
            p=np.argmin(s)
            
            #print(p)
            q=(list(self.clfs))[p]
            r=q.predict(features)
            #print(r)
            gq=np.array(r)
            
            #gq=gq.reshape(-1,1)
            #print(gq)
            #gf=(gq!=lab).astype(int)
            #print(gf)
            gf = np.array([int(i) for i in (gq!=lab)])
            #print(gf.shape)
            #gf=gf.reshape(-1,1)
            #print(gf)
            #print(c)
            ta=np.dot(gf,c)
            #print(ta)
            ta=np.sum(ta)
            
            tf=0.5*np.log((1-ta)/ta)
            tf=np.sum(tf)
            #print(tf)
            #print(c.shape)
            #print(lab.shape)
            #print((np.exp(-1*tf)).shape)
            c[lab==gf]=c[lab==gf]*np.exp(-1*tf)
            c[lab!=gf]=c[lab!=gf]*np.exp(tf)
            #print(c)
            c=c/np.sum(c)
            self.clfs_picked.append(q)
            self.betas.append(tf)
           

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)


class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # TODO: implement "train"
        ############################################################
        #b=[]
        #for _ in range(len(labels)):
            #b.append(0.5)
        b = [0.5 for i in range(len(features))]
        c = np.array(b)
        #c = c.reshape(-1,1)
        #print(c.shape)

        #lab=np.array(labels).reshape(-1,1)
        lab = np.array(labels)
        h=[]
        for i in range(len(self.clfs)):
            j=(list(self.clfs))[i].predict(features)
            h.append(j)

#         h.append(j)
        g=np.array(h)
        #print(g)
        #f=np.zeros([len(labels),1])
        f=np.zeros(len(labels))
        for _ in range(self.T):
            z=((lab+1)/2 -c)/(c*(1-c))
#             print(z.shape)
            #print(lab.shape)
            #print(g[0].shape)
            v=c*(1-c)
            
            u=[]
            for k in range(len(g)):
                #print(z.T.shape)
                
                uk=(z-g[k])**2
                #print(uk)
                u.append(uk)

            u=np.array(u)
            s=np.dot(u,v)
            p=np.argmin(s)
            #print(p)
            q=(list(self.clfs))[p]
            r=q.predict(features)
            r=np.array(r)
            f=f+(0.5*r)
            c=1/(1+(np.exp(-2*f)))
            self.clfs_picked.append(q)
            self.betas.append(0.5)

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)

