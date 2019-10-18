
# Input: numpy vector theta of d rows, 1 column
# numpy vector x of d rows, 1 column
# Output: label (+1 or -1)
class Predict():
  def __init__(self, theta, x):
    self.theta = theta
    self.x = x

  def predict(self):
      import numpy as np
      self.theta.shape = self.theta.shape[0]
      self.x.shape = self.x.shape[0]
      if np.dot(self.theta, self.x) > 0:
          label = 1.0
      else:
          label = -1.0
      return label
