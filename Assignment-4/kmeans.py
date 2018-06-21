import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape
        #print('n,d:',N,D)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        
        
        
        import random
        mea=random.sample(x.tolist(),self.n_cluster)
        mek=np.array(mea)
        
        pt = np.zeros([len(x), self.n_cluster])
        j = 0
        for i in range(self.max_iter):
        
            ar = np.array(x)
            tor = np.zeros([len(x), self.n_cluster])
            pu = {}
            
            
            for z, b in enumerate(mek):
                re = ar - np.array(b)
                tor[:, z] = np.linalg.norm(re, axis=1)
            index = np.argmin(tor, axis=1)
            
            pt= np.zeros([len(x), self.n_cluster])
            for mz in range(len(x)):
                pt[mz,index[mz]] = 1
            
            for j in range(len(mek)):
                pu[j] = (ar[np.where(index == j), :]).tolist()         
            
            
            ph = np.zeros([len(x),self.n_cluster])
            for mz in range(len(x)):
                for mz_z in range(self.n_cluster):
                    ph[mz,mz_z] = np.sum((mek[mz_z]-x[mz])**2)
            j_new = (np.sum(pt*ph))/len(x)
            
            
            if(np.absolute(j-j_new) <= self.e):
                
                se = (np.array(mek),index.reshape([len(x)]),i)
                
                return(se)
            j = j_new
            
            mek_n=[]
            sy = sorted(pu.keys())
            
            for zt in sy:
                jh=np.array(pu[zt][0])
                jm=np.mean(jh,axis=0)
                jl=jm.tolist()
                mek_n.append(jl)
                
            if set([tuple(le) for le in mek_n]) == set([tuple(lk) for lk in mek]):
                se = (np.array(mek),index.reshape([len(x)]),i)
                
                return(se)
            mek = mek_n
        
        
        se = (np.array(mek),index.reshape([len(x)]),self.max_iter)
        
        return(se)
            
            
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        cla=[]
        centroid_labels=[]
        k = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, z, rp = k.fit(x)
        
        
        for counter in range(len(centroids)):
            py = {}
            for t in np.unique(y):
                py[t]=0
            for u in range(len(x)):
                if counter == z[u]: 
                    py[y[u]] = py[y[u]]+1
            ab=sorted(py,key=py.get,reverse=True)[0]
            cla.append(ab)
        
        centroid_labels = np.array(cla)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            #'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        
        #calculate nearest centroid
        cj=[]
        r=[]
     
        for i in range(len(x)):
            ct=[]
            for j in range(len(self.centroids)):    # TAKING LOOP FOR EACH CENTROID AND DATA POINT
                dist=np.linalg.norm(x[i]-self.centroids[j])
                ct.append(dist)
            co=np.array(ct)
            p=np.argmin(co) 
            q=self.centroid_labels[p]  # CORRELATING ARGMIN  P WITH CENTROID LABELS
            r.append(q)
        s=np.array(r)      # OUTPUT ARE THE LABELS PREDICTED
        return s
            
        # DONOT CHANGE CODE BELOW THIS LINE
