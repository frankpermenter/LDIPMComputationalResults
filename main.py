import numpy as np
from RandomDataSet import RandomDataSet
from LDIPM import longstep

def DoTest(num_vars, num_ineqs, rank_W):
    problems = RandomDataSet(30, num_var = num_vars, num_ineq = num_ineqs, rank_W = rank_W, identity_initialization = False, seed = 1)

    iters_log = []
    iters_dual = []
    iters_primal = []

    max_iters = 30
    target_duality_gap = 1e-3

    while (problems.next()):
        P, q, A, u = problems.GetProblemData()
        problem = problems.GetProblemName()

        if problems.InitialGuessAvailable():
            initial_guess = problems.GetInitialGuess()
            
            iters_log.append(longstep(P, q, -A, u).solve(iters = max_iters,  step_type = 'log', target_duality_gap = target_duality_gap)) 

            iters_dual.append(longstep(P, q, -A, u).solve(iters = max_iters,  step_type = 'dual', target_duality_gap = target_duality_gap))

            iters_primal.append(longstep(P, q, -A, u).solve(iters = max_iters, step_type = 'primal', target_duality_gap = target_duality_gap))

    format_for_paper = False
    if format_for_paper:
        format_string_data = "NumVars: {0:3d}, NumIneqs: {1:3d}, RankW: {2:3d}   " 
        format_string_results = "Log: {0:1.1f}, Dual: {1:1.1f}, Primal: {2:1.1f}   "
    else:
        format_string_data = "{0:3d} & {1:3d} & {2:3d} &   " 
        format_string_results = "{0:1.1f} & {1:1.1f} & {2:1.1f} \\\\   "

    string = format_string_data.format(num_vars, num_ineqs, rank_W)
    string = string + format_string_results.format(np.mean(iters_log), np.mean(iters_dual), np.mean(iters_primal))
    print(string)

def DoTestHelper(n):
    DoTest(num_ineqs = 200*n, rank_W = 0*n, num_vars = 100*n)
    DoTest(num_ineqs = 200*n, rank_W = 50*n, num_vars = 100*n)
    DoTest(num_ineqs = 200*n, rank_W = 100*n, num_vars = 100*n)
    DoTest(num_ineqs = 100*n, rank_W = 50*n, num_vars = 100*n)
    DoTest(num_ineqs = 150*n, rank_W = 50*n, num_vars = 100*n)
    DoTest(num_ineqs = 200*n, rank_W = 50*n, num_vars = 100*n)

DoTestHelper(1)
DoTestHelper(10)
