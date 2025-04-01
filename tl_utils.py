# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:03:05 2024

@author: yifei
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref, 2) # Relative L2 error
ml2e = lambda yest, yref : 1/yref.shape[0]*spl.norm(yest - yref, 2) # Mean L2 error
infe = lambda yest, yref : spl.norm(yest - yref, np.inf) # L-Infinity error

def rbf_kernel_2d(X1, X2, var, gamma_1, gamma_2):
        
    """
    Computes the 2D Radial Basis Function (RBF) kernel matrix between two sets of input points.

    The RBF kernel has the form:
        K(x, x') = var * exp(-||x - x'||^2), 
    where anisotropy is introduced by scaling each coordinate axis with separate length scales (gamma_1, gamma_2).

    Parameters
    ----------
    X1 : ndarray of shape (n1, 2)
        First set of 2D input points.
    X2 : ndarray of shape (n2, 2)
        Second set of 2D input points.
    var : float
        Variance parameter (kernel amplitude).
    gamma_1 : float
        Length scale for the first (x) coordinate.
    gamma_2 : float
        Length scale for the second (y) coordinate.

    Returns
    -------
    K : ndarray of shape (n1, n2) Kernel matrix 
    """
    
    n1, n2 = X1.shape[0], X2.shape[0]
    
    X1_, X2_ = np.copy(X1), np.copy(X2)
    
    X1_[:, 0] = X1[:, 0] / gamma_1
    X1_[:, 1] = X1[:, 1] / gamma_2
    
    X2_[:, 0] = X2[:, 0] / gamma_1
    X2_[:, 1] = X2[:, 1] / gamma_2
    
    n1sq = np.sum(X1_**2, axis=1).reshape(n1, 1)
    n2sq = np.sum(X2_**2, axis=1).reshape(1, n2)
    
    D = n1sq + n2sq - 2 * np.matmul(X1_, X2_.T) #(n1,n2)
    K = var*np.exp(-D)
    
    return K  

def darcy_solver_implicit_with_source(Nx, Nt, dx, dt, Kx, u0, ul, ur, f, Q):
    
    '''
        Solves the 1D transient Darcy flow equation using implicit finite difference scheme.
         ∂u/∂t = ∂/∂x (K(x) ∂u/∂x) + f(x, t) + Q(x, t),

        Parameters
        ----------
        Nx : int
            Number of spatial grid cells (Nx + 1 total grid points/nodes).
        Nt : int
            Number of time steps.
        dx : float
            Spatial grid spacing.
        dt : float
            Time step size.
        Kx : ndarray of shape (Nx + 1,)
            Spatially varying hydraulic conductivity at each grid point.
        u0 : ndarray of shape (Nx + 1,)
            Initial pressure/head distribution at time t = 0.
        ul : float
            Dirichlet boundary condition at the left boundary (x = 0).
        ur : float
            Dirichlet boundary condition at the right boundary (x = L).
        f : ndarray of shape (Nt + 1, Nx + 1)
            Source term defined over time and space.
        Q : ndarray of shape (Nt + 1, Nx + 1)
            Additional volumetric source/sink term (e.g., wells), defined over time and space.

        Returns
        -------
        u : ndarray of shape (Nt + 1, Nx + 1) of pressure head
    '''
    
    a = dt / dx**2
    A = np.zeros((Nx + 1, Nx + 1))

    for i in range(1, Nx):
        A[i, i-1] = -a * (2 * Kx[i] * Kx[i - 1]) / (Kx[i] + Kx[i - 1])
        A[i, i+1] = -a * (2 * Kx[i] * Kx[i + 1]) / (Kx[i] + Kx[i + 1])
        A[i, i] = 1 - A[i, i-1] - A[i, i+1]

    A[0, 0] = 1
    A[1, 0] = 0
    A[-1, -1] = 1
    A[-2, -1] = 0

    b = np.zeros(Nx + 1)
    b[1] = a * (2 * Kx[1] * Kx[0]) * ul / (Kx[1] + Kx[0])
    b[-2] = a * (2 * Kx[Nx - 1] * Kx[Nx]) * ur / (Kx[Nx - 1] + Kx[Nx])

    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = u0
    u[0, 0] = ul
    u[0, -1] = ur

    for n in range(Nt):
        source_term = f[n, :] + Q[n, :]
        source_term[0] = 0
        source_term[-1] = 0

        rhs = u[n, :] + dt * source_term + b
        rhs[0] = ul
        rhs[-1] = ur
        u[n + 1, :] = npl.solve(A, rhs)

    return u 

def darcy_solver_implicit(K, Nt, Nx, ul, ur, u0, dt, dx):
    
    """
    Solves the 1D transient Darcy equation using an implicit finite difference scheme 
    with harmonic averaging of heterogeneous hydraulic conductivity.

    The equation solved is:
        ∂u/∂t = ∂/∂x (K(x) ∂u/∂x),

    where:
        - u(x, t) is the hydraulic head or pressure,
        - K(x) is the spatially varying hydraulic conductivity.

    Dirichlet boundary conditions are enforced at both ends:
        u(0, t) = ul,  u(L, t) = ur,
    with the initial condition:
        u(x, 0) = u0(x).

    Parameters
    ----------
    K : ndarray of shape (Nx,)
        Hydraulic conductivity values at each spatial grid point.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points.
    ul : float
        Dirichlet boundary condition at the left boundary (x = 0).
    ur : float
        Dirichlet boundary condition at the right boundary (x = L).
    u0 : ndarray of shape (Nx,)
        Initial hydraulic head at t = 0.
    dt : float
        Time step size.
    dx : float
        Spatial grid spacing.

    Returns
    -------
    u : ndarray of shape (Nt, Nx)
        Time evolution of the solution, where u[n, i] is the hydraulic head at time step n and spatial index i.
    """

    u = np.zeros((Nt, Nx))
    u[0, :] = u0
    uFD = u0*np.ones(Nx)
    
    A = np.zeros((Nx, Nx))
    A[0, 0] = 1
    A[-1, -1] = 1
    CN = 2 * dt / dx**2
    for i in range(1, Nx - 1):
        A[i, i] = CN * (K[i + 1] * K[i] / (K[i + 1] + K[i]) + K[i] * K[i - 1] / (K[i] + K[i - 1])) + 1
        A[i, i - 1] = -CN * K[i] * K[i - 1] / (K[i] + K[i - 1])
        A[i, i + 1] = -CN * K[i + 1] * K[i] / (K[i + 1] + K[i])

    for l in range(1, Nt):
        b = uFD.copy()
        b[0] = ul
        b[-1] = ur
        uFD = np.linalg.solve(A, b)
        u[l, :] = uFD

    return u

def darcy_residual(meanK, PhiK, xiK, meanU, PhiU, eta,  Nt, Nx, dt, dx):
    
    """
    Computes the residual of the 1D transient Darcy equation using FD discretization,
    with permeability and pressure fields reconstructed from KL expansions, i.e.,
        K(x) ≈ meanK(x) + Φ_K(x) @ ξ_K,
        u(x, t) ≈ meanU(x, t) + Φ_U(x, t) @ η.
    Residuals are computed at interior space-time points using backward Euler in time and harmonic averaging for K.
    Note that time index starts from the second index to the last
    and space index starts from the second to the second last (cuz there are second-order derivatives)
    If not, there will be very large residuals at the boundary

    Parameters
    ----------
    meanK : ndarray of shape (Nx,)
        Mean of the permeability field.
    PhiK : ndarray of shape (Nx, rK)
        eigenvectors of the permeability field.
    xiK : ndarray of shape (rK,)
        KL coefficients for the permeability field.
    meanU : ndarray of shape (Nt * Nx,)
        Mean of the pressure field, flattened in row-major order (time-major).
    PhiU : ndarray of shape (Nt * Nx, rU)
        eigenvectors for the pressure field.
    eta : ndarray of shape (rU,)
        KL coefficients for the pressure field.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial points.
    dt : float
        Time step size.
    dx : float
        Spatial grid spacing.

    Returns
    -------
    residual : ndarray of shape (Nt, Nx) Discrete residual of the Darcy equation
    """

    K = meanK + PhiK@xiK
    u = (meanU + PhiU@eta).reshape((Nt, Nx))
    LHS = np.zeros((Nt, Nx))
    RHS = np.zeros((Nt, Nx))
    residual = np.zeros((Nt, Nx))

    LHS[1:, 1:-1] = (u[1:, 1:-1] - u[:-1, 1:-1])/dt 
    K_eff = 2 * K[1:] * K[:-1] / (K[1:] + K[:-1]) # effective conductivity
    RHS[1:,1:-1] = (K_eff[1:] * (u[1:, 2:] - u[1:, 1:-1]) # right flux 
                   - K_eff[:-1]* (u[1:, 1:-1] - u[1:, :-2])) / dx**2 # left flux
    residual = LHS - RHS
    
    return residual

def get_eigenpairs(cov, eps = np.sqrt(np.finfo(float).eps)):
    
    """
        Computes the eigendecomposition of a symmetric covariance matrix with regularization.

        Parameters
        ----------
        cov : ndarray of shape (N, N)
            Symmetric covariance matrix (e.g., empirical covariance from Monte Carlo ensembles).
        eps : float, optional
            Small positive value added to the diagonal of `cov` to ensure positive definiteness.
            Default is machine epsilon for float precision.

        Returns
        -------
        Phi_sqrt : ndarray of shape (N, N)
            Matrix of square-root-scaled eigenvectors. Columns are ordered according to descending eigenvalues.
            Phi_sqrt = Phi @ sqrt(diag(Lambda)), where `cov = Phi @ diag(Lambda) @ Phi.T`.
        Lambda_sorted : ndarray of shape (N,)
            Eigenvalues of the covariance matrix in descending order.
    """

    Lambda, Phi = spl.eigh(cov + eps * np.eye(cov.shape[0]))  #(N, ) (N, N)
    return (Phi.real @ np.diag(np.sqrt(np.abs(Lambda))))[:, ::-1], Lambda[::-1]

########## Plotting utility ###########

def plot_field_2d(XX, TT, u, title, **kwargs):

    figsize = kwargs.get('figsize', (4, 4))
    dpi = kwargs.get('dpi', 300)
    fontsize = kwargs.get('fontsize', 12)
    cmap = kwargs.get('cmap', 'turbo')

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    c = ax.pcolormesh(XX, TT, u, cmap = cmap)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel(r'$x$', fontsize=fontsize)
    ax.set_ylabel(r'$t$', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(c, cax=cax)
    plt.tight_layout()
    plt.show()
     
    return

def plot_sol(xr, u_ref, u_pred, Nrt, label = 'source', title = None):

    c = ['#FF5575', '#FFD36A', '#6299FF']
    t = [Nrt//50, Nrt//5, Nrt -1]
    
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    ax.plot(xr, u_ref[t[0], :], color=c[0], linewidth=2.5, label='Ref, $t = t_1$') 
    ax.plot(xr, u_pred[t[0], :], 'o', color=c[0], markersize=6, markeredgewidth=2, mfc='none', label='Pred, $t = t_1$')
    ax.plot(xr, u_ref[t[1], :], color=c[1], linewidth=2.5, label='Ref, $t = t_2$')
    ax.plot(xr, u_pred[t[1], :], 'o', color=c[1], markersize=6, markeredgewidth=2, mfc='none', label='Pred, $t = t_2$')
    ax.plot(xr, u_ref[t[2], :], color=c[2], linewidth=2.5, label='Ref, $t = t_3$')
    ax.plot(xr, u_pred[t[2], :], 'o', color=c[2], markersize=6, markeredgewidth=2, mfc='none', label='Pred, $t = t_3$')
    ax.set_xlabel(r'$X$', fontsize=14)
    ax.set_ylabel(r'$h$,' + 'source', fontsize=14)
    ax.set_xlim(-0.01, 1.01)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.show()

    return fig, ax

def plot_loss(train_loss, test_loss, epochs, num_print, save_fig_to):
    
    t = np.arange(0, epochs, num_print)
    fig = plt.figure(constrained_layout=False, figsize=(4, 4), dpi = 300)
    ax = fig.add_subplot()
    ax.plot(t, train_loss, color='blue', label='Training Loss')
    ax.plot(t, test_loss, color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss',  fontsize = 16)
    ax.set_xlabel('Epochs', fontsize = 16)
    ax.legend(loc='upper right', fontsize = 14)
    fig.tight_layout()
    fig.savefig(save_fig_to + 'loss.png')
    
    return fig, ax

## Code to cpmpute finite differene derivative with batch dimensions

# def darcy_residual(meanK, PhiK, xiK, meanU, PhiU, eta,  Nt, Nx, dt, dx):
    
#     K = meanK + PhiK@xiK
#     u = (meanU + PhiU@eta).reshape((Nt, Nx))
#     residual = np.zeros((Nt, Nx))
    
#     for l in range(1, Nt):
#         for i in range(1, Nx - 1):
#             LHS = (u[l, i] - u[l-1, i]) / dt
#             K_iphalf = 2 * K[i+1] * K[i] / (K[i+1] + K[i])
#             K_imhalf = 2 * K[i] * K[i-1] / (K[i] + K[i-1])
#             RHS = (K_iphalf * (u[l, i+1] - u[l, i]) - K_imhalf * (u[l, i] - u[l, i-1])) / dx**2
#             residual[l, i] = LHS - RHS
    
#     return residual

## the derivative of the mean u wrt t 
# d1tdmeanU = np.zeros((Nrt, Nrx))
# d1tdmeanU[1:, 1:-1] = (meanU_rs[1:, 1:-1] - meanU_rs[:-1, 1:-1]) / dt

# # the derivative of u eigenfunctions wrt t
# d1tdPhiU = np.zeros((Nrt, Nrx, Nuxi))
# d1tdPhiU[1:, 1:-1, :] = (PhiU_rs[1:, 1:-1, :] - PhiU_rs[:-1, 1:-1, :]) / dt

# # the second derivative of the mean u times mean K wrt x
# def d2xdmeanUK(xi_K):
#     Nbatch = xi_K.shape[0]
#     K = np.einsum('ij, nj->ni', PhiK[:,:NKxi], xi_K) + meanK  #(Nbatch, Nrx-1) 
#     K_eff = (2 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])).reshape(Nbatch, 1, Nrx-1)
#     d2xdmeanUK_ = np.zeros((Nbatch, Nrt, Nrx))
#     # d2xdmeanUK_[:, :, :-1] = (np.tile(K_eff.reshape(Nbatch, 1, Nrx - 1), (1, Nrt, 1)) * 
#     #                             (meanU_rs[:, 1:] - meanU_rs[:, :-1]))/ (dx * dx)
#     d2xdmeanUK_[:, 1:, 1:-1] = (K_eff[:, :, 1:]*(meanU_rs[np.newaxis, 1:, 2:] - meanU_rs[np.newaxis, 1:, 1:-1]) - 
#                                 K_eff[:, :, :-1]*(meanU_rs[np.newaxis, 1:, 1:-1] - meanU_rs[np.newaxis, 1:, :-2])) / dx**2
    
#     return d2xdmeanUK_

# def d2xdPhiUK(xi_K):
#     Nbatch = xi_K.shape[0] 
#     K = np.einsum('ij, nj->ni', PhiK[:,:NKxi], xi_K) + meanK 
#     K_eff = (2 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])).reshape(Nbatch, 1, Nrx-1, 1)
#     d2xdPhiUK_ = np.zeros((Nbatch, Nrt, Nrx, Nuxi))
#     d2xdPhiUK_[:, 1:, 1:-1, :] = (K_eff[:, :, 1:, :]*(PhiU_rs[np.newaxis, 1:, 2:, :] - PhiU_rs[np.newaxis, 1:, 1:-1, :]) -
#                                     K_eff[:, :, :-1, :]*(PhiU_rs[np.newaxis, 1:, 1:-1, :] - PhiU_rs[np.newaxis, 1:, :-2, :]) )  / dx**2
#     return d2xdPhiUK_

# def residual(xi_K, xi_U):
#     A = d1tdPhiU  - d2xdPhiUK(xi_K)
#     b = d1tdmeanU - d2xdmeanUK(xi_K)
#     return np.einsum('nijk,nk->nij', A, xi_U) + b
