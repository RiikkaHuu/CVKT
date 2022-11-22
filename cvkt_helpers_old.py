import numpy as np
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent
from pymanopt import Problem

"""
    Copyright (C) 2019  Riikka Huusari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Helper functions for CVKT algorithm in "cvkt.py".

-----

 - Fixed a small bug (added phi=phi.T) on 01/2020  (bug had only been present in this tidied version for publication)

"""


def centering_mtrx(n):

    return np.eye(n) - (1 / n) * np.ones((n, n))


def alignment_cvkt(phi, M, U):

    phi = phi.T

    [m, n] = phi.shape
    [mp, _] = U.shape
    p = int(mp/m)

    C = centering_mtrx(n*p)

    mat = np.dot(np.dot(np.kron(phi, np.eye(p)), C),
                 np.dot(M, np.dot(C, np.kron(np.transpose(phi), np.eye(p)))))
    mat2 = np.dot(np.kron(phi, np.eye(p)), np.dot(C, np.kron(np.transpose(phi), np.eye(p))))

    upstairs = np.trace(np.dot(np.transpose(U), np.dot(mat, U)))
    downstairs = np.trace(np.dot(np.dot(np.transpose(U), np.dot(mat2, U)),
                                   np.dot(np.transpose(U), np.dot(mat2, U))))

    return upstairs / np.sqrt(downstairs)


def derivative_alignment_cvkt(phi, M, U):

    phi = phi.T

    [m, n] = phi.shape
    [mp, _] = U.shape
    p = int(mp/m)

    C = centering_mtrx(n*p)

    mat = np.dot(np.dot(np.kron(phi, np.eye(p)), C),
                  np.dot(M, np.dot(C, np.kron(np.transpose(phi), np.eye(p)))))
    mat2 = np.dot(np.kron(phi, np.eye(p)), np.dot(C, np.kron(np.transpose(phi), np.eye(p))))

    upstairstrace = np.trace(np.dot(np.transpose(U), np.dot(mat, U)))
    downstairstrace = np.trace(np.dot(np.dot(np.transpose(U), np.dot(mat2, U)),
                                   np.dot(np.transpose(U), np.dot(mat2, U))))

    dupstairs = 2*np.dot(mat, U)

    ddownstairstrace = 4*np.dot(np.dot(mat2, U), np.dot(np.dot(np.transpose(U), mat2), U))
    ddownstairs = ddownstairstrace / (2 * np.sqrt(downstairstrace))

    derivative = (dupstairs * np.sqrt(downstairstrace) - upstairstrace * ddownstairs) / downstairstrace

    return derivative


def solve_cvkt(Ktarget, Psi, rank):

    [_, a] = Psi.shape

    # initialize D

    Uinit = np.random.rand(a, rank)
    Uinit = Uinit / np.linalg.norm(Uinit, ord='fro')  # so that the init is in manifold

    # then solve it

    manifold = Sphere(a, rank)

    def Uloss(U):

        align = alignment_cvkt(Psi, Ktarget, U)

        return - align  # we want to maximize alignment, Pymanopt minimizes everything

    def Ugrad(U):

        deriv = derivative_alignment_cvkt(Psi, Ktarget, U)

        return - deriv

    problem = Problem(manifold=manifold, cost=Uloss, verbosity=0, egrad=Ugrad)

    problemsolver = SteepestDescent(maxiter=200)

    U = problemsolver.solve(problem, x=Uinit)

    return U


def get_nystrom_features(kernel, m_param):

    # copied and modified from my previous Nyström things

    n = kernel.shape[0]
    m = int(np.ceil(m_param * n))

    # there is random selection on which samples contribute on approximation, done by re-ordering kernel matrix
    if m < n:
        order = np.random.permutation(n)
        inverse_order = np.argsort(order)

        kernel = kernel[np.ix_(order, order)]

    E = kernel[:, 0:m]
    W = E[0:m, :]
    Ue, Va, _ = np.linalg.svd(W)
    vak = Va[0:m]
    inVa = np.diag(vak ** (-0.5))
    U = np.dot(E, np.dot(Ue[:, 0:m], inVa))

    if m < n:
        U = U[inverse_order, :]

    return U


def get_where_elems_in_second_list_are_in_first(list1, list2):

    inds = []
    for jjj in list2:
        inds.append(np.argwhere(np.array(list1) == jjj)[0][0])
    return inds


def mean_imputation(N, availability):

    M = np.copy(N)

    available = np.where(availability == 1)[0]
    missing = np.where(availability == 0)[0]

    mean = np.mean(M[np.ix_(available, available)])

    M[missing, :] = mean
    M[:, missing] = mean

    return M


def uniform_rescaling_on_known_part_match_means(Mknown, Mtarget, availability):

    indx_known = np.where(availability==1)[0]

    # make uniform (linear) rescaling on the data values in target matrix such that the range of values in known part
    # is the same as range of values in Mknown

    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range

    # from [a b] to [c d]

    c = np.min(Mknown[np.ix_(indx_known, indx_known)])
    d = np.max(Mknown[np.ix_(indx_known, indx_known)])

    a = np.min(Mtarget[np.ix_(indx_known, indx_known)])
    b = np.max(Mtarget[np.ix_(indx_known, indx_known)])

    Mtarget = c + ((d-c)/(b-a))*(Mtarget-a)

    # after that match the means

    mean1 = np.mean(Mknown[np.ix_(indx_known, indx_known)])
    mean2 = np.mean(Mtarget[np.ix_(indx_known, indx_known)])
    Mtarget = Mtarget-(mean2-mean1)

    return Mtarget
