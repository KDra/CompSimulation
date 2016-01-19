
# coding: utf-8

# # Self Consistent Field Theory - Lab 1

# In[1]:

from IPython.core.display import HTML
css_file = 'https://raw.githubusercontent.com/ngcm/training-public/master/ipython_notebook_styles/ngcmstyle.css'
HTML(url=css_file)


# In[5]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)
from scipy.integrate import quad


# In Hartree-Fock theory the time-independent Schrödinger equation
# 
# \begin{equation}
#   H \psi = E \psi
# \end{equation}
# 
# is solved (approximately), where $H$ is the Hamiltonian operator, $\psi$ the wavefunction, and $E$ the energy. The theory assumes that the wave function depends on the locations of the electrons ${\bf r}_i$, where eg ${\bf r}_1$ is the spatial location of the first electron, and that the wave function can be written as the *Slater determinant*
# 
# \begin{equation}
#   \psi = \frac{1}{\sqrt{N}} \begin{vmatrix} \chi_1({\bf r}_1) & \chi_2({\bf r}_1) & \dots &\chi_N({\bf r}_1) \\ \chi_1({\bf r}_2) & \chi_2({\bf r}_2) & \dots &\chi_N({\bf r}_2) \\ \vdots & \vdots & \ddots & \vdots \\ \chi_1({\bf r}_N) & \chi_2({\bf r}_N) & \dots &\chi_N({\bf r}_N) \end{vmatrix}
# \end{equation}

# After some considerable theoretical work, the Hartree-Fock equations are
# 
# \begin{equation}
#   F({\bf x}_1) \chi_i({\bf x}_1) = \epsilon_i \chi_i({\bf x}_1),
# \end{equation}
# 
# where $F$ is the *Fock* operator
# 
# \begin{equation}
#   F({\bf x}_1) = H({\bf x}_1) + \sum_j \left( J_j({\bf x}_1) - K_j({\bf x}_1) \right),
# \end{equation}
# 
# with $J$ being the Coulomb operator
# 
# \begin{equation}
#   J_j({\bf x}_1) = \int \text{d} {\bf x}_2 \, \frac{| \chi_j ({\bf x}_2) |^2}{r_{12}}
# \end{equation}
# 
# and $K$ is the exchange operator
# 
# \begin{equation}
#   K_j({\bf x}_1) \chi_i({\bf x}_1) = \left[ \int \text{d} {\bf x}_2 \, \chi^*_j ({\bf x}_2) \frac{1}{r_{12}} \chi_i ({\bf x}_2) \right] \chi_j({\bf x}_1).
# \end{equation}
# 
# In the above $r_{12}$ is the distance between the first and second electrons, $r_{12} = \| {\bf x}_1 - {\bf x}_2 \|$.

# As the Hamiltonian operator $H$ contains a second partial derivative ($H = -\frac{1}{2} \nabla^2 + \dots$) this is a set of integro-differential equations, which is painful to solve numerically (see [this review](http://dx.doi.org/10.1016/j.cpc.2012.09.033) for an example). Instead, as with Finite Elements, it's better to write the orbitals $\chi$ in terms of a function basis, as
# 
# \begin{equation}
#   \chi_i = \sum_{\mu=1}^K C_{i\mu} \tilde{\chi}_{\mu}.
# \end{equation}
# 
# Here the function basis is *global*: there is one expansion that holds over all of space.
# 
# This leads to the Hartree-Fock-Roothaan equations
# 
# \begin{equation}
#   {\bf F} {\bf C} = {\bf S} {\bf C} \epsilon
# \end{equation}
# 
# where all of the terms are matrices representing the operators. Written in more detail we have
# 
# \begin{equation}
#   \sum_{\nu} F_{\mu\nu} C_{\nu i} = \epsilon_i \sum_{\nu} S_{\mu \nu} C_{\nu i}
# \end{equation}
# 
# where the matrices are
# 
# \begin{align}
#   S_{\mu \nu} &= \int \text{d} {\bf x}_1 \, \tilde{\chi}^*_{\mu}({\bf x}_1) \tilde{\chi}_{\nu}({\bf x}_1), \\
#   F_{\mu \nu} &= \int \text{d} {\bf x}_1 \, \tilde{\chi}^*_{\mu}({\bf x}_1) F({\bf x}_1) \tilde{\chi}_{\nu}({\bf x}_1).
# \end{align}
# 
# For later purposes we define the *density matrix* ${\bf D}$ as
# 
# \begin{equation}
#   D_{\mu \nu} = \sum_{j=1}^{N_{\text{electrons}}/2} 2 C_{\mu j} C_{\nu j},
# \end{equation}
# 
# from which we write the Fock matrix as
# 
# \begin{equation}
#   F_{\mu \nu} = H_{\mu \nu} + \sum_{\alpha} \sum_{\beta} \left( G_{\mu \nu \alpha \beta} - \frac{1}{2} G_{\mu \beta \alpha \nu} \right) D_{\alpha \beta},
# \end{equation}
# 
# where $H$ is the one-electron operator in the function basis
# 
# \begin{equation}
#   H_{\mu \nu} = \int \text{d}{\bf x}_1 \, \chi_{\mu}({\bf x}_1) \left( - \frac{1}{2} \nabla^2 \right) \chi_{\nu}({\bf x}_1) + \sum_a \int \text{d}{\bf x}_1 \, \chi_{\mu}({\bf x}_1) \frac{Z_a}{|{\bf R}_a - {\bf r}_1|} \chi_{\nu}({\bf x}_1)
# \end{equation}
# 
# and $G$ is the two-electron operator in the function basis
# 
# \begin{equation}
#   G_{\mu \nu \alpha \beta} = \int \text{d}{\bf x}_1 \, \text{d}{\bf x}_2 \, \chi_{\mu}({\bf x}_1) \chi_{\nu}({\bf x}_2) \frac{1}{r_{12}} \chi_{\alpha}({\bf x}_1) \chi_{\beta}({\bf x}_2).
# \end{equation}
# 
# Finally, the total energy $E$ is given by
# 
# \begin{equation}
#   E = \frac{1}{2} \sum_{\mu=1}^N \sum_{\nu=1}^N D_{\mu\nu} \left( H_{\mu\nu} + F_{\mu\nu} \right) + V_{\text{nn}},
# \end{equation}
# 
# where $V_{\text{nn}}$ is the nucleon-nucleon interaction energy
# 
# \begin{equation}
#   V_{\text{nn}} = \sum_{a} \sum_{b} \frac{Z_a Z_b}{\| {\bf R}_a - {\bf R}_b \|}.
# \end{equation}

# ## Self-consistent field solution procedure

# This is an iterative prcedure, so must start from some initial guess.
# 
# 1. Calculate all one- and two-electron integrals, $H$ and $G$.
# 2. Generate starting guess for the $C$ (molecular orbital [MO]) coefficients.
# 3. Form the density matrix $D$.
# 4. Form the Fock matrix $F$ from the core (one-electron) integrals $H$ plus the density matrix $D$ times the two-electron integrals $G$.
# 5. Diagonalize the Fock matrix $F$. The eigenvectors contain the new MO coefficients.
# 6. Form the new density matrix $D$. If sufficiently close to the old matrix we are done; otherwise, return to step 4.

# The first step is difficult, so we will assume here that the elements of the $H$ matrix and $G$ tensor are given.
# 
# The crucial point is that all steps must be performed in the right basis, and whilst the basis changes between steps, the transformation matrix stays fixed. Given the overlap matrix $S$ between the basis functions, the transformation matrix $X$ is given by $U \Lambda^{-1/2} U^*$, where $U$ and $\Lambda$ are the eigenvectors and eigenvalues of $S$ respectively.

# ### Code

# Write a function that, given $S$, computes the transformation matrix $X = U \Lambda^{-1/2} U^*$ (using `numpy.linalg.eig`).

# In[40]:

def tranformMat(S):
    w, v = np.linalg.eig(S)
    w = w**(-1/2)
    return np.dot(np.dot(v, w), v.conj().T)


# Write a function that, given $C$ and the number of electrons, computes the density matrix $D$, where
# 
# \begin{equation}
#   D_{\mu \nu} = \sum_{j=1}^{N_{\text{electrons}}/2} 2 C_{\mu j} C_{\nu j}.
# \end{equation}

# In[24]:

def denseMat(C, Nelec):
    D = np.zeros_like(C)
    n = int(Nelec/2.0)
    for mu in np.arange(len(C)):
        for nu in np.arange(len(C)):
            D[mu, nu] = np.sum(2*C[mu, :n]*C[mu, :n])
    return D


# Write a function that, given $H, G$ and $D$, computes the Fock matrix $F$, where
# 
# 
# \begin{equation}
#   F_{\mu \nu} = H_{\mu \nu} + \sum_{\alpha} \sum_{\beta} \left( G_{\mu \nu \alpha \beta} - \frac{1}{2} G_{\mu \beta \alpha \nu} \right) D_{\alpha \beta}.
# \end{equation}

# In[25]:

def fock(H, G, D):
    F = np.zeros_like(D)
    n = len(D)
    for mu in np.arange(n):
        for nu in np.arange(n):
            F[mu, nu] = H[mu, nu] + np.sum(np.dot(G[mu, nu, :, :] - 0.5 * G[mu, :, :, nu], D))
    return F


# Write a function that, given $F$, uses the `numpy.linalg.eigh` function to extract the eigenvalues and eigenvectors. It should return the orbital energies (the eigenvalues in order) and the new orbital coefficients ($X V$, where $V$ is the matrix of eigenvectors). It should compute $F' = X^* F X$ in the transformed basis, compute its eigenvalues $\epsilon$ and eigenvectors $V$, and hence get the new coefficients $X V$.

# In[42]:

def coef(F, X):
    F = np.dot(X.conj().T, F)
    F = np.dot(F, X)
    w, v = np.linalg.eigh(F)
    return w, np.dot(X, v)


# Write a function that, given $X, H, G$, and a guess for $C$ with its associated density matrix $D$, returns the new density matrix, new basis coefficients, and orbital energies.

# In[45]:

def dens(X, H, G, C, D, Nelec):
    F = fock(H, G, D)
    print(F)
    eps, C = coef(F, X)
    D = denseMat(C, Nelec)
    return D, C, eps


# Write a function that, given $D, H, F$ and $V_{\text{nn}}$, returns the total energy of the configuration.

# \begin{equation}
#   E = \frac{1}{2} \sum_{\mu=1}^N \sum_{\nu=1}^N D_{\mu\nu} \left( H_{\mu\nu} + F_{\mu\nu} \right) + V_{\text{nn}},
# \end{equation}

# In[36]:

def energy(D, H, F, Vnn):
    return 0.5 * np.sum(np.dot(D, H + F)) + Vnn


# Write a function that, given $S, H, G, V_{\text{nn}}$ and a guess for $C$, iterates the Hartree-Fock method until it converges to a certain tolerance. It should print the total energy of the configuration.

# In[38]:

def iterate(S, H, G, C, Vnn, Nelec, tol=1e-4):
    en_old = 0
    en_new = np.inf
    while abs(en_old-en_new) > tol:
        en_old = en_new
        X = tranformMat(S)
        D = denseMat(C, Nelec)
        D, C, eps = dens(X, H, G, C, D, Nelec)
        en_new = energy(D, H, F, Vnn)
    print("The total Energy of the configuration is {}".format(en_new))


# ### Example

# A two-electron system would be $\text{He} - \text{H}_+$ - one Helium and one Hydrogen, with an electron missing. The required input data is:

# In[3]:

Nelectrons = 2
S = np.array([[1.0, 0.434311], [0.434311, 1.0]])
H = np.array([[-1.559058, -1.111004], [-1.111004, -2.49499]])
G = np.array([[[[ 0.77460594,  0.27894304],[ 0.27894304,  0.52338927]],
                  [[ 0.27894304,  0.14063907],[ 0.14063907,  0.34321967]]],
                 [[[ 0.27894304,  0.14063907],[ 0.14063907,  0.34321967]],
                  [[ 0.52338927,  0.34321967],[ 0.34321967,  1.05571294]]]])
Vnn = 1.3668670357


# Check that your algorithm works: the total energy should be approximately $-2.626$ (Hartrees), and the initial guess can be pure zeros. It should take around 15 iterations.

# In[46]:

C = np.zeros_like(H)
iterate(S, H, G, C, Vnn, Nelectrons, tol=1e-3)


# In[ ]:



