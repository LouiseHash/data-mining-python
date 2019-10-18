# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)

class Nearestneighbor():
  def __init__(self, X=[], y=[],z=[]):
    self.X = X
    self.y = y
    self.z=z

  def nearestneighbor(self):
    import numpy as np
    import numpy.linalg as la
    n = len(self.y)
    d = len(self.X[0])
    c = 0
    b = la.norm(self.z - self.X[0])
    for t in range(n):
      X_t = np.zeros((d, 1))
      for i in range(d):
        X_t[i][0] = self.X[t][i]
      if la.norm(self.z - X_t) < b:
        c = t
        b = la.norm(self.z - X_t)
    label = self.y[c][0]
    return label


