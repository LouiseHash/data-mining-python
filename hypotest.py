# Input: numpy vector a of scalar values, with k rows, 1 column
# numpy vector b of scalar values, with k rows, 1 column
# scalar alpha
# Output: reject (0 or 1)

class Hypotest():

  def __init__(self, a, b, alpha):
    self.a = a
    self.b = b
    self.alpha = alpha

  def hypotest(self):
      import math
      k = len(self.a)
      mu_1 = 0
      mu_2 = 0
      sigma_1 = 0
      sigma_2 = 0

      for i in range(k):
          mu_1 += self.a[i]
      mu_1 = (1 / float(k)) * mu_1

      for i in range(k):
          mu_2 += self.b[i]
      mu_2 = (1 / float(k)) * mu_2

      for i in range(k):
          sigma_1 += (self.a[i] - mu_1) ** 2
      sigma_1 = (1 / float(k)) * sigma_1

      for i in range(k):
          sigma_2 += (self.b[i] - mu_2) ** 2
      sigma_2 = (1 / float(k)) * sigma_2

      x = ((mu_1 - mu_2) * k ** 0.5) / (sigma_1 + sigma_2) ** 0.5

      v = math.ceil(((sigma_1 + sigma_2) ** 2 * (k - 1)) / (sigma_1 ** 2 + sigma_2 ** 2))

      from scipy.stats import t
      x_alpha_v = t.ppf(1 - self.alpha, v)

      if x > x_alpha_v:
          reject = 1
      else:
          reject = 0
      return reject
