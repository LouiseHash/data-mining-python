# Input: scalar q
# numpy vector mu_positive of d rows, 1 column
# numpy vector mu_negative of d rows, 1 column
# scalar sigma2_positive
# scalar sigma2_negative
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)

class Probcpredict():
  def __init__(self,q,mu_positive, mu_negative, sigma2_positive, sigma2_negative, z):
    self.q = q
    self.mu_positive = mu_positive
    self.mu_negative=mu_negative
    self.sigma2_positive=sigma2_positive
    self.sigma2_negative=sigma2_negative
    self.z=z

  def probcpredict(self):
    import numpy as np
    import math
    import numpy.linalg as la
    d = len(self.mu_positive)
    if math.log(np.true_divide(self.q, 1 - self.q)) - np.true_divide(d, 2) * math.log(
            np.true_divide(self.sigma2_positive, self.sigma2_negative)) - np.true_divide(1,
                                                                                         2 * self.sigma2_positive) * np.square(
      la.norm(self.z - self.mu_positive)) + np.true_divide(1, 2 * self.sigma2_negative) * np.square(
      la.norm(self.z - self.mu_negative)) > 0:
      label = 1.0
    else:
      label = -1.0
    return label
