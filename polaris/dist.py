import numpy as np

class DistributionField:
    """A DistributionField represents many fluorophore distributions. It 
    consists of an array of Distributions."""
    def __init__(self, sh_arr=None, f_arr=None):
        if f_arr is not None and sh_arr is not None:
            print("Warning: sh_arr and f_arr are redundant.")
        elif f_arr is None:
            self.sh_arr = sh_arr
            self.f_arr = None
        elif sh_arr is None:
            self.sh_arr = None
            self.f_arr = f_arr

    def calc_f_arr(self, B):
        self.f_arr = np.einsum('ij,klmj->klmi', B, self.sh_arr)

    def make_positive(self, B, max_l=None):
        for i in np.ndindex(self.sh_arr.shape[:2]):
            d = Distribution(self.sh_arr[i])
            d.make_positive(B, max_l=max_l)
            self.sh_arr[i] = d.sh
            
class Distribution:
    """A Distribution represents a fluorophore distribution. It has redundant
    representations in the angular domain (orientation distribution function)
    and the angular frequency domain (spherical harmonic coefficients).

    """
    def __init__(self, sh=None, f=None):
        if f is None:
            self.sh = sh
        if sh is None:
            self.f = f

    def make_positive(self, B, max_l=None):
        # Setup convex problem
        from cvxopt import matrix, solvers
        N = B.shape[1]
        M = B.shape[0]
        P = matrix(2*np.identity(N), tc='d')
        q = matrix(-2*self.sh, tc='d')
        G = matrix(-B, tc='d')
        h = matrix(np.zeros(M), tc='d')
        if max_l is None:
            sol = solvers.qp(P, q, G, h)
        else:
            from polaris import util
            J = util.maxl2maxj(max_l)
            A = matrix(np.identity(N)[N-J:,:], tc='d')
            b = matrix(np.zeros(J), tc='d')
            sol = solvers.qp(P, q, G, h, A, b)
        self.sh = np.array(sol['x']).flatten()
