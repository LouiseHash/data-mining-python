
import numpy as np
from probclearn import Probclearn
from nearestneighbor import Nearestneighbor
from probcpredict import Probcpredict

#case 3_1
np.set_printoptions(precision=4)
X = np.array([[-3, 2],
                  [-2, 1.5],
                  [-1, 1],
                  [0, 0.5],
                  [1, 0]])
y = np.array([[1], [1], [1], [-1], [-1]])
z = np.array([[1], [-2]])
pl=Probclearn(X,y)
q,mu_pos,mu_neg,sigma2_pos,sigma2_neg = pl.probclearn()
pp=Probcpredict(q,mu_pos,mu_neg,sigma2_pos,sigma2_neg, z)
nn=Nearestneighbor(X,y,z)
print("Predict result of probcpredict:",pp.probcpredict())
print("Predict result of nearestneighbor:",nn.nearestneighbor())



