import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None
        
    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the dim of feature to be splitted

        self.feature_uniq_split = None # the feature to be splitted


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
                        
            '''
            branches: C x B array, 
                          C is the number of classes,
                          B is the number of branches
                          it stores the number of corresponding training examples
            '''
            ########################################################
            # TODO: compute the conditional entropy
            ########################################################
            
            a=np.array(branches)
            
            c=a.copy()
            p=np.sum(a,axis=0)
            c[c==0]=1
            d=np.log(c/p)
            e=a/p
            f=e*d
            g=np.sum(f,axis=0)*(-1)
            h=p/sum(p)
            l=np.sum(g*(h))
            return l
        
        th=[]
        for idx_dim in range(len(self.features[0])):
        ############################################################
        # TODO: compare each split using conditional entropy
        #       find the best split
        ############################################################
        
            fe=np.array(self.features).transpose()
            fa=fe[idx_dim]
            fg=set(fa)
            ft=list(fg)
            n=len(ft)
            lb=set(self.labels)
            lt=list(lb)
            m=len(lt)
            le=len(self.features)
            rt=np.zeros((m,n))
            
            dt=dict()
            j=0
            for af in ft:
                dt[af]=j
                j=j+1
                
            i=0
            dc=dict()
            for fc in lt:
                dc[fc]=i
                i=i+1
                
            
            for index in range(le):
                re=dt[self.features[index][idx_dim]]
                rf=dc[self.labels[index]]
                
                
                rt[rf][re]+=1
            th.append(conditional_entropy(rt))
            
        if th==[]:
            self.dim_split=None
            self.feature_uniq_split=None
            self.splittable=False
            return
        
        yu=min(th)
        hj=th.index(yu)
        ou=fe[hj]
        ot=set(ou)
        self.feature_uniq_split = list(ot)
        self.dim_split=hj
        ############################################################
        # TODO: split the node, add child nodes
        ############################################################

        for w in ot:
            kl=[]
            kj=[]
            for j,dh in enumerate(self.features):
                if dh[hj]==w:
                    kj.append(self.labels[j])
                    kl.append(dh)
                    
            
            kl=np.delete(kl,hj,1)
            km=kl.tolist()
            sd=set(kj)
            ln=len(sd)
            
            self.children.append(TreeNode(km,kj,ln))
            
        
        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()
            
        return

    
    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            #print('New')
            #print(feature)
            #print(self.dim_split)
            #print(feature)
            #print(feature[self.dim_split])
            #print(self.feature_uniq_split.index(feature[self.dim_split]))
            #print(self.children)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split+1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



