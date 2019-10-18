
import numpy as np
from perceptron import Perceptron
from predict import Predict
from svm import Svm
from probclearn import Probclearn
from nearestneighbor import Nearestneighbor
from probcpredict import Probcpredict
from kfoldcv import Kfoldcv
from bootstrapping import Bootstrapping
from hypotest import Hypotest

#case 2_1
np.set_printoptions(precision=4)
X = np.array([[-3, 2],
[-2, 1.5],
[-1, 1],
[0, 0.5],
[1, 0]])
y = np.array([[1], [1], [1], [-1], [-1]])
pe=Perceptron(10,X,y)
theta_perceptron, num = pe.perceptron()
sv=Svm(X,y)
theta_svm = sv.svm()
pr=Predict(theta_perceptron,np.array([[1], [-2]]))
print("From preceptron algorithm, the theta is:",theta_perceptron,"using direction [[1], [-2]], it will be predicted as:",pr.predict())
pr2=Predict(theta_svm,np.array([[1], [-2]]))
print("From svm, the theta is:",theta_svm,"using direction [[1], [-2]], it will be predicted as:",pr2.predict())


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

#case 4_1

np.set_printoptions(precision=4)
X = np.array([[-3, 2],
[-2, 1.5],
[-1, 1],
[0, 0.5],
[1, 0],
[2, 2],
[-0.5, -1],
[0.5, 0]])
y = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1]])
kf=Kfoldcv(2,X,y)
print("When k equals to 2, the accuracy of kfold is:",kf.kfoldcv())
bs=Bootstrapping(5,X,y)
np.random.seed(26)
print("When B equals to 5, the accuracy of bootstrapping is:",bs.bootstrapping())

# for Hypotest
a = np.array([[0.09],[0.08],[0.15],[0.11],[0.13]])
b = np.array([[0.10],[0.12],[0.14],[0.13],[0.13]])
ht=Hypotest(a,b,0.05)
print("When alpha equals to 0.05, the result of hypotest is:",ht.hypotest())


