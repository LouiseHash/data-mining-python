
# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column


class Kfoldcv():

  def __init__(self, k, X, y):
    self.k = k
    self.X = X
    self.y = y

  def kfoldcv(self):
    from probcpredict import Probcpredict
    from probclearn import Probclearn
    import math
    import numpy as np
    n = len(self.y)
    d = len(self.X[0])
    z = np.zeros((self.k, 1))
    for i in range(self.k):
      T = set(range(int(math.floor(float(n) * i / float(self.k))), int(math.floor((float(n) * (i + 1) / float(self.k)) - 1) + 1)))
      S = set(range(0, n)) - T
      X_train = np.zeros((len(S), d))
      y_train = np.zeros((len(S), 1))
      index = []
      for x in S:
        index.append(x)
      for t in range(len(S)):
        X_train[t] = self.X[index[t]]
        y_train[t] = self.y[index[t]]
      pc=Probclearn(X_train, y_train)
      q, mu_plus, mu_minus, sigma_plus, sigma_minus = pc.probclearn()

      for t in T:
        x = self.X[t].reshape(self.X[t].shape[0], 1)
        pp = Probcpredict(q, mu_plus, mu_minus, sigma_plus, sigma_minus, x)
        # x=np.zeros((d,1))
        # for i in range(d):
        # x[i]=X[t][i]
        if self.y[t] != pp.probcpredict():
          z[i] = z[i] + 1
      z[i] = z[i] / float(len(T))
      sum=0
      for i in range(self.k):
        sum=z[i]+sum
      accuracy=sum/float(self.k)
    return accuracy[0]



