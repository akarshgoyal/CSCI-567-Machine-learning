from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
      - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
### base case
  for j in range(S):
    q=O[1]
    alpha[j,0]=pi[j]*B[j,q]
    
    
  #print (alpha)


  qe=np.zeros(S)
  for t in range(1,N):
    for j in range(S):
#       for i in range(S):
        q=O[t]
#         qe[j]=qe[j]+A[i,j]*alpha[i,q]
#       alpha[j,t]=B[j,q]*qe[j]
        hu=np.dot(A[:,j].T,alpha[:,t-1].T)
        alpha[j,t] = np.multiply(B[j,q],hu.T)
 
  #print (alpha)

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N  = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  for j in range(S):
    beta[j][N-1]=1

  qa=np.zeros(S)

  for t in range(N-1,0,-1):
    for i in range(S):
      for j in range(S):
        q=O[t]
        beta[i][t-1]=beta[i][t-1]+beta[j,t]*A[i,j]*B[j,q]
      
#   for t in range(N-1, 0, -1):
#         for i in range(S):
#             for j in range(S):
#                 beta[i][t-1] += beta[j][t] * A[i][j] * B[j][O[t]]
  
  #print(beta)
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################

  prob=np.sum(alpha[:,-1])
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  for i in range(len(pi)):
    q=O[0]
    prob=prob+beta[i,0]*pi[i]*B[i,q]
    
  #print(prob)
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
       
  sr = {}
  ej=[]
  fkh={}
  viterbi_alg=[]
  viterbi_alg.append(fkh)
  
  for j in range(len(A)):
        ej.append(j)
  
  q=O[0]
  for ds in ej:
      sr[ds] = [ds]
      viterbi_alg[0][ds] = np.multiply(B[ds,q],pi[ds])
      
     

        
  for jk in range(1, len(O)):
      path_n = {}
      viterbi_alg.append({})           
      for ds in ej:
          uj=[]
          for dso in ej:
              q=O[jk]
              uj.append(viterbi_alg[jk-1][dso] * A[dso][ds] * B[ds][q])
          vu=np.array(uj)
          gds=np.max(vu)
          gfh=np.argmax(vu)
          viterbi_alg[jk][ds] = gds
          path_n[ds] = [ds] + sr[gfh]  
      sr = path_n
  lam = 0    
  if len(O)!=1:
      lam = jk

#   vi=np.array(viterbi_alg[lam])
#   #print(viterbi_alg)
#   gfh=np.argmax(vi)
#   path=sr[gfh]
#   path=path[::-1] 
  (gds, gfh) = max((viterbi_alg[lam][dso], dso) for dso in ej)
  #for dso in ej:
  #(gds,gfh)=int(max(viterbi_alg[lam][dso],dso))
  path=sr[gfh]
  path=path[::-1]
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
