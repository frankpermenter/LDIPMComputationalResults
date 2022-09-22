import numpy as np

def NormalizeRows(M):
    for i in range(0, M.shape[0]):
        M[i, :] = M[i, :]/np.linalg.norm(M[i, :])
    return M

class RandomDataSet:
    def __init__(self, num_instances, num_var = 40, num_ineq = 100, rank_W = -1, seed = -1, identity_initialization = False):
        if (seed != -1):
            np.random.seed(seed)

        self.num_instances = num_instances
        self.instance_number = 0
        self.initial_guess_available = True
        self.identity_initialization = identity_initialization
        self.num_var = num_var
        self.num_ineq = num_ineq
        self.rank_W = rank_W
        if rank_W == -1:
            self.rank_W = max(num_var  - num_ineq, 0)

    def next(self):
        self.instance_number = self.instance_number + 1
        if (self.instance_number > self.num_instances): 
            return False
        num_var = self.num_var
        rank_P = self.rank_W
        num_ineq = self.num_ineq

        self.A = NormalizeRows(np.matrix( np.random.randn(num_ineq, num_var)  ))

        x = np.matrix(np.random.randn(num_var, 1))   
        e = np.matrix(np.ones( (num_ineq, 1) ))

        if (self.identity_initialization):
            s0 = e
            l0 = e
        else:
            eps = 0.1
            s0 = e + eps*np.abs(np.random.randn(num_ineq, 1))
            l0 = e + eps*np.abs(np.random.randn(num_ineq, 1))

        self.u =  s0  +  self.A * x

        if rank_P > 0:
            Psqrt = NormalizeRows(np.random.randn(num_var, rank_P))
            self.P =  Psqrt @ Psqrt.transpose() 
        else:
            self.P  = np.random.randn(num_var, num_var) * 0

        self.q = -(self.A.transpose() * l0 + self.P * x)

        self.initial_guess = {}
        self.initial_guess['x'] = x
        self.initial_guess['s'] = s0
        self.initial_guess['l'] = l0
        return True

    def GetProblemData(self):
        return self.P, self.q, self.A, self.u

    def GetProblemName(self):
        return str(self.instance_number)

    def InitialGuessAvailable(self):
        return self.initial_guess_available
    def GetInitialGuess(self):
        return self.initial_guess
