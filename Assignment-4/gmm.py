import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None
        self.lg= None

    def fit(self, x):
        
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k
            
            k_means = KMeans(self.n_cluster,self.max_iter,self.e)           
            self.means, memb, i = k_means.fit(x)                      # means, membership and no. of updates are assigned
            
            gamm=np.zeros((N,self.n_cluster))
            
            gamm[np.arange(N),memb] = 1                             # finding the responsibility
            
            gamma=gamm.T                                           # taking transpose as gamma is assigned
            gamma_sum=np.zeros(self.n_cluster)                    # initializing N_k
            
            #print(gamma.shape)
            
            self.variances=np.empty((self.n_cluster,len(x.T),len(x.T)))
            self.pi_k=np.zeros(self.n_cluster)
            
            i_ii=[]                  
            for item in self.means:                 #finding x-u for each cluster and assigning in x_u
                i_i=[]
                for i in x:
                    i_item=i-item
                    i_itemi=i_item.tolist()
                    i_i.append(i_itemi)
                i_ii.append(i_i)
            x_u=np.array(i_ii)
            
            
            gamma_sum=np.sum(gamma,axis=1)
            covariance=np.zeros((self.n_cluster,D,D))
            #print(temp.shape)
            for i in range(self.n_cluster): 
#                 gamma_sum[i] = temp[i]
                #gamma_sum[i]=np.sum(gamma[i],axis=1)               # finding N_k using gamma
                #print(gamma[i].shape)
                temp = np.multiply(gamma[i].reshape(-1,1),x_u[i])
                covariance[i]=np.dot(temp.T,x_u[i])/gamma_sum[i]    # finding variance using equation (10)
                
            self.pi_k=gamma_sum/len(x)                 # finding p(z=k)
            self.variances = covariance
            

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
             #   'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE
            
        

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            
            import random
            mea=random.sample(x.tolist(),self.n_cluster)
            self.means=np.array(mea)
            self.pi_k=np.multiply(np.ones(self.n_cluster),1/self.n_cluster)
            self.variances=np.array([np.identity(len(x.T))]*self.n_cluster)
            
            #raise Exception('Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        #else:
            #raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
 
        
        


        logl=self.compute_log_likelihood(x)              #Algorithm step-4
        logl = 0
        br=0 
        for _ in range(self.max_iter):              #Step-5
            print(br)
            br=br+1                                   #for finding the updates
            
            #self.compute_log_likelihood(x)
            #print(self.lg)
            i_ii=[]                  
            for item in self.means:                 #finding x-u for each cluster and assigning in x_u
                i_i=[]
                for i in x:
                    i_item=i-item
                    i_itemi=i_item.tolist()
                    i_i.append(i_itemi)
                i_ii.append(i_i)
            x_u=np.array(i_ii)

            pir_d=np.power((np.pi)*2,len(x.T))                 #pir_d is (2pi)^d
            
            det=np.zeros(self.n_cluster)
            inv_v=np.zeros(self.n_cluster)

            for i,t in enumerate(self.means):
                det[i]=np.linalg.det(self.variances[i])
                inv_v[i]=1/(np.power(np.multiply(pir_d,det[i]),0.5))   # 1/(((2pi)^d)*variance)^0.5
            
            log_l=np.zeros(len(x))
            
            gamma_l=[]
            gae_l=[]
            
            hy=[]
            
            

            for i in range(self.n_cluster):                              #running loop for each cluster
                a=np.dot(x_u[i],np.linalg.pinv(self.variances[i]))
                b=np.multiply(x_u[i],a)                                # taking dot of inverse of variance and x-u for each cluster and then multiplying with x-u
                b=np.sum(b,axis=1)
                c=np.multiply(-0.5,b)                                  # multiply with -0.5 above line
                d=np.exp(c)                                            # take exponent
                pxz_k=np.multiply(inv_v[i],d)                             # multiply with 1/(((2pi)^d)*variance)^0.5
                #print(pxz_k.shape)
                pxy_k=pxz_k.tolist()
                hy.append(pxy_k)
            ho=np.array(hy)
            
            
            l_l=np.multiply(self.pi_k.reshape(-1,1),ho)                    # multiply with p(z=k)
            #print(l_l.shape)
            #print(l_l)
            
            l_la=np.sum(l_l,axis=0)
            
            #print(l_la)
            gamma=np.divide(l_l,l_la+1e-40)
            gamma_sum=np.sum(gamma,axis=1)        # convert to numpy array result of line-144 
            
            #print(gamma.shape)
            gamma_u=gamma_sum.reshape(-1,1)
            #print(gamma_sum.shape)
            
            self.means=np.divide(np.dot(gamma,x),gamma_u+1e-40)
            self.pi_k=np.divide(gamma_sum,len(x))
            
            #print(self.pi_k)
            #print(gamma_u.shape)
            
            for i in range(self.n_cluster):
                
                fh=np.sum((np.multiply(np.dot(gamma[i],x_u[i]),x_u[i])),axis=0)/(gamma_u[i]+1e-40)
                fh=fh.reshape(-1,1)
                #print(fh.shape)
                #print(self.variances[i].shape)
                self.variances[i]=np.dot(fh,fh.T)  # find variance using (10)
                #print(self.variances[i].shape)
                
            #print(self.variances.shape)     
            l_new=self.compute_log_likelihood(x)       # find new log_likelihood
            if(logl-l_new<self.e):
                break
                
            else:
                logl=l_new
        
        #print(br)
        return br                    # return the number of updates
    
    
                
            
        
        
        #raise Exception('Implement fit function (filename: gmm.py)')
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        #if (self.means is None):
            #raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        
        p=np.random.choice(self.n_cluster,N)
        
        g=[]
        for py in p:
            f=np.random.multivariate_normal(self.means[py],self.variances[py],1)
            fh=np.ravel(f)
            g.append(fh)
        
        yu=np.array(g)
        return yu    
        
        
        
        #raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        
        for gh in range(self.n_cluster):
            while (np.linalg.matrix_rank(self.variances[gh])!=len(x.T)):
                
                self.variances[gh]=self.variances[gh]+0.001*np.identity(len(x.T))
            
        
        
        i_ii=[]
        for item in self.means:   #finding x-u for each cluster and assigning in x_u
            i_i=[]
            for i in x:
                i_item=i-item
                i_itemi=i_item.tolist()
                i_i.append(i_itemi)
            i_ii.append(i_i)
        x_u=np.array(i_ii)
        
        pir_d=np.power((np.pi)*2,len(x.T))         #pir_d is (2pi)^d
        
        
        
        for i,t in enumerate(self.means):
            det=np.linalg.det(self.variances[i])
        
        inv_v=1/(np.power(np.multiply(pir_d,det),0.5))   # 1/(((2pi)^d)*variance)^0.5
        
        
        log_l=np.zeros(len(x))
        
        #print(self.pi_k)
        hj=[]
        for i in range(self.n_cluster):                              #running loop for each cluster
            a=np.dot(x_u[i],np.linalg.pinv(self.variances[i]))
            
            b=np.multiply(x_u[i],a)                                # taking dot of inverse of variance and x-u for each cluster and then multiplying with x-u
            b=np.sum(b,axis=1)
            #print(b)
            c=np.multiply(-0.5,b)                                  # multiply with -0.5 above line
            d=np.exp(c)                                            # take exponent
            
            pxz_k=np.multiply(inv_v,d)                             # multiply with 1/(((2pi)^d)*variance)^0.5
            #print(pxz_k.shape)
            pxy_k=pxz_k.tolist()
            hj.append(pxy_k)
        ho=np.array(hj)
            
        #print(self.pi_k.shape)
        #print(ho)
        l_l=np.multiply(self.pi_k.reshape(-1,1),ho)                    # multiply with p(z=k)
        #print(l_l.shape)
            #print(l_l)
            
        l_la=np.sum(l_l,axis=0)
            #print(l_la)
        log_l=l_la                                        # add all cluster vectors
            
        #print(log_l)   
        log_ll=np.float128(np.log(log_l+1e-40))                                       # take log of result in line-237
        #print(log_ll)
        
        log_lik=np.sum(log_ll)                                     # find sum of all element in the vector found from line-238 
        #print(log_lik)
        self.lg=log_l                                              # assign to global variable to be used by fit function
        #print(self.lg)
        #print(log_lik)
        return float(log_lik)                                           # return log likelihood

        
        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
