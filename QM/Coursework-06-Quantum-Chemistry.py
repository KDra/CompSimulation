
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)
rcParams['contour.negative_linestyle'] = 'solid'
from scipy.integrate import quad
from numpy import pi
from numpy.linalg import norm


R_O = np.array([0.0, 1.809*np.cos(104.52/180.0*np.pi/2.0), 0.0])
R_H1 = np.array([-1.809*np.sin(104.52/180.0*np.pi/2.0), 0.0, 0.0])
R_H2 = np.array([+1.809*np.sin(104.52/180.0*np.pi/2.0), 0.0, 0.0])
scale = norm(R_O-R_H1) *1.5

Vnn = 8.90770810

S = np.array([[ 1.       ,  0.2367039,  0.       ,  0.       , -0.       ,
         0.0500137,  0.0500137],
       [ 0.2367039,  1.       ,  0.       ,  0.       , -0.       ,
         0.4539953,  0.4539953],
       [ 0.       ,  0.       ,  1.       ,  0.       ,  0.       ,
         0.       ,  0.       ],
       [ 0.       ,  0.       ,  0.       ,  1.       ,  0.       ,
         0.2927386, -0.2927386],
       [-0.       , -0.       ,  0.       ,  0.       ,  1.       ,
         0.2455507,  0.2455507],
       [ 0.0500137,  0.4539953,  0.       ,  0.2927386,  0.2455507,
         1.       ,  0.2510021],
       [ 0.0500137,  0.4539953,  0.       , -0.2927386,  0.2455507,
         0.2510021,  1.       ]])

H = np.array([[ -3.26850823e+01,  -7.60432270e+00,   0.00000000e+00,
          0.00000000e+00,  -1.86797000e-02,  -1.61960350e+00,
         -1.61960350e+00],
       [ -7.60432270e+00,  -9.30206280e+00,   0.00000000e+00,
          0.00000000e+00,  -2.22159800e-01,  -3.54321070e+00,
         -3.54321070e+00],
       [  0.00000000e+00,   0.00000000e+00,  -7.43083560e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         -7.56702220e+00,   0.00000000e+00,  -1.89085610e+00,
          1.89085610e+00],
       [ -1.86797000e-02,  -2.22159800e-01,   0.00000000e+00,
          0.00000000e+00,  -7.52665570e+00,  -1.65878930e+00,
         -1.65878930e+00],
       [ -1.61960350e+00,  -3.54321070e+00,   0.00000000e+00,
         -1.89085610e+00,  -1.65878930e+00,  -4.95649010e+00,
         -1.56026360e+00],
       [ -1.61960350e+00,  -3.54321070e+00,   0.00000000e+00,
          1.89085610e+00,  -1.65878930e+00,  -1.56026360e+00,
         -4.95649010e+00]])

Nelectrons = 10

G=np.fromfile('./H2O-two-electron.dat')
G = np.reshape(G,(7,7,7,7));



def tranformMat(S):
    """
    Returns the transformation matrix X given the separation matrix S.
    
    Internal variables:
    w       eigenvalues of S
    v       eigenvectors of S
    Inputs:
    S       matrix of double
            Separation of atoms           
    Output:
    X       matrix of double
            Transformation matrix
    """
    w, v = np.linalg.eig(S)
    w = np.diag(w**(-1/2))
    X = np.dot(np.dot(v, w), v.conj().T)
    return X

def denseMat(C, Nelec):
    """
    Return the density matrix D given the total number of electrons and the 
    coefficient matrix C used to obtain the molecular orbitals.
    """
    assert np.shape(C)[0] == np.shape(C)[1],\
        "The coefficient matrix should be square."""
    
    D = np.zeros_like(C)
    n = int(Nelec/2.0)
    
    for mu in np.arange(len(C)):
        for nu in np.arange(len(C)):
            for j in np.arange(n):
                D[mu, nu] += 2*C[mu, j]*C[nu, j]
    return D

def fock(H, G, D):
    """
    Return the Fock 'operator' F given the Hamiltonian 'operator' H, the two
    electron interaction 'operator' G and the density matrix D.
    """
    F = np.zeros_like(D)
    n = len(D)
    for mu in np.arange(n):
        for nu in np.arange(n):
            F[mu, nu] = H[mu, nu]
            for a in np.arange(n):
                for b in np.arange(n):
                    F[mu, nu] += (G[mu, nu, a, b] - 0.5 * G[mu, b, a, nu]) * D[a,b]
    return F

def coef(F, X):
    """
    Returns the orbital energies and a new approximation of the coefficient
    matrix C by doing an eigen decomposition on the transformed Fock operator.
    """
    F = np.dot(X.conj().T, F)
    F = np.dot(F, X)
    orb_enrg, v = np.linalg.eigh(F)
    C = np.dot(X, v)
    return orb_enrg, C

def dens(X, H, G, C, D, Nelec):
    """
    Returns a new approximation of the density matrix D by computing the
    the Fock operator and then using it to compute a new coefficient matrix C.
    The function also returns the improved guess for C and the orbital energies
    in 'eps'.
    """
    F = fock(H, G, D)
    eps, C = coef(F, X)
    D = denseMat(C, Nelec)
    return D, C, eps

def energy(D, H, F, Vnn):
    """
    Returns the total energy of the system, given an orbital configuration.
    """
    return 0.5 * np.sum(D * (H + F)) + Vnn

def iterate(S, H, G, C, Vnn, Nelec, tol=1e-4):
    """
    Returns the matrix of coefficients for the molecular orbitals in C for the 
    lowest energy configuration of the system. This is computed by looking at
    the changes in the density matrix to avoid an additional function evaluation
    at each step. The methods are nevertheless equivalent.
    """
    D_old = denseMat(C, Nelec)
    D_new = D_old + 10
    i=0
    while np.max(np.abs(D_old-D_new)) > tol:
        D_old = D_new.copy()
        X = tranformMat(S)
        D_new, C, eps = dens(X, H, G, C, D_old, Nelec)
        i+=1
    F = fock(H, G, D_new)
    en = energy(D_new, H, F, Vnn)
    print("The total Energy of the configuration is {} at iter = {}".format(en, i))
    return C

# Vectors to contain all the gaussian coefficients
c = np.array([[0.444635, 0.535328, 0.154329], [0.700115, 0.399513, -0.0999672], [0.391957, 0.607684, 0.1559163]])
alpha = np.array([[0.109818, 0.405771, 2.22766], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203]])

# Compute the orbital coefficient matrix
C = np.zeros_like(S)
C = iterate(S, H, G, C, Vnn, Nelectrons, tol=1e-10)

def basisH(r, c, alpha, zeta = 1.24**2):
    """
    Function to compute the gaussians given a set of coefficients c and alpha,
    with a default value of zeta included for hydrogen.
    """
    assert len(np.shape(c)) == 1, "Ensure that c is a vector"
    assert len(np.shape(alpha)) == 1, "Ensure that alpha is a vector"
    assert len(c) == len(alpha), "Vectors alpha and c must be of equal length"
    assert np.shape(r)[0] == np.shape(r)[1], "Ensure that r is a square matrix"
    assert type(zeta) == float, "Zeta must be of type float"
    
    basis = np.zeros_like(r)
    for i in np.arange(len(c)):
        basis += c[i]*(2 * alpha[i] * zeta / np.pi)**0.75 * np.exp(-alpha[i] * zeta * r**2)
    return basis

def basisO(r, c, alpha, orb=0):
    """
    Function to compute the gaussians given a set of coefficients c and alpha,
    with default values of zeta for the s1 and other orbitals harcoded for Oxygen.
    The preffered orbital is chosen by selecting an integer between 0 and 4 in
    the 'orb' variable, where the values correspond to enumerated orbital names
    from the innermost at 1s - 0 to the outermost at 2p - 2, 3, 4.
    """
    assert np.shape(c) == np.shape(alpha),\
    "Gaussian coefficient matrices alpha and c must be of equal length"
    assert np.shape(r)[0] == np.shape(r)[1], "Ensure that r is a square matrix"
    
    zeta_s1 = 7.66**2
    zeta = 2.25**2
    if orb == 0: 
        return basisH(r, c[0], alpha[0], zeta = zeta_s1)
    elif orb == 1: 
        return basisH(r, c[1], alpha[1], zeta = zeta)
    elif orb < 5 and orb >1:
        return basisH(r, c[2], alpha[2], zeta = zeta)
    else:
        raise ValueError("Error! The number of orbitals for oxygen must be between 0 and 5")

def coord():
    """
    Computes the real coordinates of the orbitals of water and it returns the 
    values in a 3D matrix where the 0th dimension points to the coordinates for
    each orbital.
    """
    # Produce a grid
    l = np.linspace(-scale, scale, 100)
    x, y = np.meshgrid(l, l)
    # Compute distances from each atom
    rO = np.sqrt((x-R_O[0])**2 + (y-R_O[1])**2)
    rH1 = np.sqrt((x-R_H1[0])**2 + (y-R_H1[1])**2)
    rH2 = np.sqrt((x-R_H2[0])**2 + (y-R_H2[1])**2)
    
    chir = np.zeros((7, len(l), len(l)))
    # Compute gaussians corresponding to oxygen
    for i in np.arange(5):
        chir[i] = basisO(rO, c, alpha, orb=i)
    # Compute gaussians corresponding to hydrogen
    chir[5] = basisH(rH1, c[0], alpha[0])
    chir[6] = basisH(rH2, c[0], alpha[0])
    # Transform gaussians to obtain orbitals
    chi = np.zeros_like(chir)
    for i in np.arange(len(C)):
        for j in np.arange(len(C)):
            chi[i] += C[i, j] * chir[j]
    return chi
"""
r = coord()

titles = ["Oxygen 1s", "Oxygen 2s", "Oxygen 2p (x)", "Oxygen 2p (y)", "Oxygen 2p (z)", "Hydrogen-1 1s", "Hydrogen-2 1s"]
ext = np.array([scale]*4)
ext[1::2] *= -1
for i in np.arange(7):
    plt.figure()
    lvl = np.linspace(np.max(r[i]), np.min(r[i]), 10)
    mplt = plt.contour(r[i], extent=ext, cmap="copper", levels=lvl)
    plt.clabel(mplt, inline=1, fontsize=12, colors='k')
    plt.title(titles[i])
"""

def tests():
    def test_transformMat():
        I = np.eye(5)
        if np.any(tranformMat(I) == I):
            return True
        else:
            return False
        
    def test_denseMat():
        C = np.eye(5)
        C[1, 3] = 1.0
        D = C.copy()*2
        D[1, 3] = D[3, 1] = 2.0
        D[1, 1] = 4.0
        D[4, 4] = 0.0
        if np.all(denseMat(np.eye(5), 10) == np.eye(5)*2) and np.all(denseMat(C, 9) == D):
            return True
        else:
            return False
        
    def test_fock():
         a = np.eye(3)
         g = g=np.array([[a]*3]*3)
         F = np.array([[ 3.5,  2.5,  2.5],[ 2.5,  3.5,  2.5],[ 2.5,  2.5,  3.5]])
         if np.all(fock(a, g, a) == F):
             return True
         else:
             return False
    
    print("Transformation operation is successful: {}".format(test_transformMat()))
    print("Density matrix operation is successful: {}".format(test_denseMat()))
    print("Fock operation is successful: {}".format(test_fock()))
        
tests()