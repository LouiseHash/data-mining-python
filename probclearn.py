# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: scalar q
# numpy vector mu_positive of d rows, 1 column
# numpy vector mu_negative of d rows, 1 column
# scalar sigma2_positive
# scalar sigma2_negative

class Probclearn():
  def __init__(self, X=[], y=[]):
    self.X = X
    self.y = y
  def probclearn(self):
    import numpy as np
    import numpy.linalg as la
    d = len(self.X[0])
    n = len(self.y)
    k_positive = 0
    k_negative = 0
    mu_positive = np.zeros((d, 1))
    mu_negative = np.zeros((d, 1))
    for t in range(n):
      X_t = np.zeros((d, 1))
      for i in range(d):
        X_t[i][0] = self.X[t][i]  ####
      if self.y[t] == 1:
        k_positive = k_positive + 1
        mu_positive = mu_positive + X_t
      else:
        k_negative = k_negative + 1
        mu_negative = mu_negative + X_t
    q = np.true_divide(k_positive, n)
    mu_positive = (np.true_divide(1, k_positive)) * (mu_positive)
    mu_negative = (np.true_divide(1, k_negative)) * (mu_negative)
    sigma2_positive = 0
    sigma2_negative = 0
    for t in range(n):
      X_t = np.zeros((d, 1))
      for i in range(d):
        X_t[i][0] = self.X[t][i]
      if self.y[t] == 1:
        sigma2_positive = sigma2_positive + np.square(la.norm(X_t - mu_positive))
      else:
        sigma2_negative = sigma2_negative + np.square(la.norm(X_t - mu_negative))
    sigma2_positive = (np.true_divide(1, d * k_positive)) * sigma2_positive
    sigma2_negative = (np.true_divide(1, d * k_negative)) * sigma2_negative
    return q, mu_positive, mu_negative, sigma2_positive, sigma2_negative


