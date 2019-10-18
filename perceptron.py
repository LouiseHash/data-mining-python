# Input: maximum number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
# number of iterations that were actually executed (iter+1)
class Perceptron():
  def __init__(self, L, X, y):
    self.L = L
    self.X = X
    self.y = y

  def perceptron(self):
      import numpy as np
      d = len(self.X[0])
      n = len(self.y)
      theta = np.zeros(d)
      for iter in range(0, self.L):
          all_points_classified_correctly = True
          for t in range(0, n):
              if self.y[t] * (theta.dot(self.X[t])) <= 0:
                  theta = theta + self.y[t] * self.X[t]
                  all_points_classified_correctly = False
          if all_points_classified_correctly:
              break
      theta = np.reshape(theta, [d, 1])
      return theta, iter + 1
