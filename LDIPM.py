import scipy.linalg 
import numpy as np

def Objective(P, q, x):
    x = np.array(x[:])
    q = np.matrix(q[:])
    val = .5 * (x.transpose() @ P @ x) + q.transpose() @ x
    return val

class longstep:
    def __init__(self, W, q, A, b):
        self.q = q.reshape((-1, 1))
        self.A = A
        self.b = b.reshape( (-1, 1))
        self.G = A.transpose() @ A  + W
        self.W = W
        self.num_ineq = A.shape[0]
        self.e = np.ones((self.num_ineq, 1))
        self.v_cached = []

    def line_search(self, r, v, dinf_bound = 1):
        d0 = self.newton_dir(r = r*1e15, v = v)[0]
        d1 = self.newton_dir(r = r, v = v)[0] - d0
        min_alpha = 0
        max_alpha = 1e15
        for i in range(0, self.num_ineq):
            temp_min = float((dinf_bound-d0[i]) / d1[i])
            temp_max = float((-dinf_bound-d0[i]) / d1[i])
            if d1[i] > 0:
                temp = temp_max
                temp_max = temp_min
                temp_min = temp
                
            if temp_min > min_alpha:
                min_alpha = temp_min
            
            if temp_max < max_alpha:
                max_alpha = temp_max

        return min_alpha < max_alpha, min_alpha, max_alpha

    def newton_dir(self, r, v):
        if (self.v_cached == [] or np.linalg.norm(self.v_cached - v) > 0):
            self.v_cached = v
            D = 0 * v * v.transpose()
            expv = 0 * v
            for i in range(0, self.num_ineq):
                D[i, i] = np.exp(v[i])
                expv[i] = np.exp(v[i]) 

            self.DA = D @ self.A
            self.Db = D @ self.b
            G = self.DA.transpose() @ self.DA  + self.W
            self.cholesky_factor, self.up_low = scipy.linalg.cho_factor(G)

        x = scipy.linalg.cho_solve( (self.cholesky_factor, self.up_low), self.DA.transpose() @ ( 2 * r) - self.q - self.DA.transpose() @ self.Db)

        s = self.DA @ x + self.Db
        d = np.add(r, -s) 
        d = d/r

        beta = 1
        norminf = np.linalg.norm(d, np.Inf)
        stepsize = 1.0/max(1.0, 1.0/(2*beta) * norminf**2)

        return d, x, self.A @ x + self.b, stepsize
    
    def log(self, x):
        eps = 1e-15
        z = x * 0 + eps
        return np.log(np.maximum(x, z))

    def Newton(self, r, v0, iters = 30, eps = 1e-3, step_type = 'log'):
        v = v0
        for i in range(0, iters):
            d, x, s, stepsize = self.newton_dir(r, v)
            if step_type == 'dual':
                    v = self.log(np.multiply(np.exp(v), 1+stepsize*d))
            elif step_type == 'primal':
                    v = -self.log( np.multiply(np.exp(-v), 1-stepsize*d))
            elif step_type == 'log':
                    v = v + d * stepsize
            else:
                raise "Invalid step-type"

            if (np.linalg.norm(d) < eps):
                break

        if (np.linalg.norm(d) > eps):
            print("FAIL")
            print(np.linalg.norm(d))
            raise

        return v, stepsize

    def gap(self, v, r):
        n = self.A.shape[0]
        e = np.ones((n, 1))
        d, x, s, stepsize = self.newton_dir(r, v)
        l = np.multiply(r, e+d)
        s = np.multiply(r, e-d)
        return np.float(l.transpose() @ s) , np.float(Objective(self.W, self.q, x)[0][0])


    def solve(self, fixed_point_iters = 1, newton_iters = 1, iters = 10, step_type = 'log', target_duality_gap = 0):
        num_ineqs = self.A.shape[0]
        progress_last = np.Inf
        v = self.e * 0 
        r = self.e
        verbose = False

        if verbose:
            print("Step-type:", step_type)
        
        for iter_cnt in range(0, iters):
            success, alpha_min, alpha_max = self.line_search(r = r, v = v, dinf_bound = .99)
            if (success):
                r = r * 1.0/np.abs(alpha_max + 1e-15) 
            else:
                print(iter_cnt)
                print(alpha_min)
                print(alpha_max)
                print("line search failed")
                raise

            final_gap = self.gap(r=r, v=v)
            d, x, s, stepsize = self.newton_dir(r, v)
            dinf = np.linalg.norm(d, np.Inf)
            if dinf < 1.0 and final_gap[0] < target_duality_gap * num_ineqs:
                return iter_cnt

            v, damping = self.Newton(r = r, v0 = v, iters = newton_iters, eps = 1e10, step_type = step_type)
            v_last = v
            format_str = '{0:10.2e}'
            format_str_int = '{0:3d}'
            if verbose:
                string = ""
                string = string + " gap "+ format_str.format(final_gap[0])
                string = string + " obj "+ format_str.format(final_gap[1])
                string = string + " dinf"+ format_str.format(dinf)

                print(string)

