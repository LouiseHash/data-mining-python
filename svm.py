# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
class Svm():
  def __init__(self,  X, y):
    self.X = X
    self.y = y

  def svm(self):
      d = len(self.X[0])
      n = len(self.y)
      import numpy as np
      import cvxopt as co
      H = np.identity(d)
      f = np.zeros(d)
      A = np.zeros((n, d))
      b = (-1) * np.full(n, 1.)
      for i in range(0, n):
          for j in range(0, d):
              A[i][j] = -np.dot(self.y[i], self.X[i][j])
      theta = np.array(
          co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'), co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
      return theta
