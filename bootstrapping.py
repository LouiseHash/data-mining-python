# Input: number of bootstraps B
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column
class Bootstrapping():
    def __init__(self, B, X, y):
        self.B = B
        self.X = X
        self.y = y

    def bootstrapping(self):
        from probcpredict import Probcpredict
        from probclearn import Probclearn
        import numpy as np
        d = len(self.X[0])
        n = len(self.y)
        z = np.zeros((self.B, 1))
        for i in range(self.B):
            u = np.zeros(n)
            S = set()
            for j in range(n):
                k = np.random.randint(0, n)
                u[j] = k
                S.add(k)
            T = set(range(0, n)) - S
            X_train = np.zeros((n, d))
            y_train = np.zeros((n, 1))
            for j in range(n):
                X_train[j] = self.X[int(u[j])]
                y_train[j] = self.y[int(u[j])]
            # print(u)
            pc = Probclearn(X_train, y_train)
            q, mu_plus, mu_minus, sigma_plus, sigma_minus = pc.probclearn()
            z[i] = 0
            for t in T:
                x = self.X[t].reshape(self.X[t].shape[0], 1)
                pp = Probcpredict(q, mu_plus, mu_minus, sigma_plus, sigma_minus, x)
                if self.y[t] != pp.probcpredict():
                    z[i] = z[i] + 1
            z[i] = z[i] / float(len(T))
            sum = 0
            for i in range(self.B):
                sum = z[i] + sum
            accuracy = sum / float(self.B)
        return accuracy[0]





