import sys
import time
from datetime import timedelta
from cvkt_helpers import *


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

This file implements Cross-View Kernel Transfer (CVKT) algorithm for completing multi-view kernel matrices, as 
introduced in [Huusari, Capponi, Villoutreix and Kadri: Kernel transfer over multiple views for missing data completion]

To use CVKT import the function "cvkt_complete", where you should give 
* K: a N*N*V numpy array, consisting of the V kernel matrices to be completed (naturally it doesn't matter with what 
  values the missing values are replaced with.)
* MID: a N*V numpy array of zeros and ones, where one at [n,v] indicates that the sample n is available at view v, 
  similarly zero at [n, v] indicates that it is missing
* rank_lvl: controls the number of columns in U that is optimized, should be number in ]0,1] (zero excluded, one 
  included), indicating the percentage of rank w.r.t the possible full rank of U. 
* approx_lvl: controls the number of samples chosen for Nyström approximations, should be number in ]0,1] (zero 
  excluded, one included), indicating the percentage of the samples to be used w.r.t the samples available in the view. 

Alternatively, it is possible to use the algorithm for completing an individual view with "cvkt_complete_one_view"

-----

 - Fixed a small bug (line 111, k_target -> K[:, :, view]) on 01/2020  (bug had only been present in this tidied version for publication)
  
"""


def cvkt_complete_one_view(K, MID, view, approx_lvl, rank_lvl, normalize=True):

    """
    :param K: array [n, n, v] containing the kernels to be completed
    :param MID: array [n, v] containing on column i the availability info for kernel i; 1 means available, 0 missing
    :param view: the view this function focuses on completing
    :param approx_lvl: Nyström approximation level for getting the features (percentage, 0.1 means 10% of full)
    :param rank_lvl: the rank level of U that is learned (percentage, 0.1 means 10% of full)
    :param normalize: if the result is normalized. CVKT SHOULD ALWAYS USE NORMALIZATION IN POST-PROCESSING! REMOVING
                      NORMALIZATION WILL MAKE CVKT RESULTS UNCOMPARABLE TO OTHER ALGORITHMS
    :return: predicted (completed) kernel matrix of view "view"
    """

    # script for one view if one can/wants to make computations parallel

    available = np.where(MID[:, view] == 1)[0]
    # missing = np.where(MID[:, view] == 0)[0]

    k_target = np.squeeze(K[np.ix_(available, available, [view])])
    n_views = K.shape[2]

    # construct the \Psi
    psi = None
    psiAll = None
    for vv in range(n_views):
        if vv != view:

            # samples that are available here in this view
            available_here = np.where(MID[:, vv] == 1)[0]

            # the available features of view vv
            feats = get_nystrom_features(np.squeeze(K[np.ix_(available_here, available_here, [vv])]), approx_lvl)

            phi = np.zeros((len(available), feats.shape[1]))
            phiAll = np.zeros((K.shape[0], feats.shape[1]))

            # indices of these elements that are also available in the view
            common_elems = list(set(available_here) & set(available))
            where_available_are_in_feats = get_where_elems_in_second_list_are_in_first(available_here, common_elems)
            where_available_should_be_in_phi = get_where_elems_in_second_list_are_in_first(available, common_elems)

            phi[where_available_should_be_in_phi, :] = feats[where_available_are_in_feats, :]

            phiAll[available_here, :] = feats

            if psi is None:

                psi = phi
                psiAll = phiAll

            else:
                psi = np.hstack((psi, phi))
                psiAll = np.hstack((psiAll, phiAll))

    # then onto the alignment optimization

    rank = int(rank_lvl * psi.shape[1])  # at most phi.shape[1]

    print("solving cvkt for view "+str(view)+"...", end="")
    sys.stdout.flush()
    ttt = time.process_time()
    U = solve_cvkt(k_target, psi, rank)
    print(" solved!  "+str(timedelta(seconds=time.process_time()-ttt)))
    sys.stdout.flush()

    # predict the full kernel
    tmp = np.dot(psiAll, U)
    pred_kernel = np.dot(tmp, tmp.T)

    if normalize:
        pred_kernel = uniform_rescaling_on_known_part_match_means(K[:, :, view], pred_kernel, MID[:, view])

    return pred_kernel


def cvkt_complete(K, MID, approx_lvl, rank_lvl, normalize=True):

    """
    :param K: array [n, n, v] containing the kernels to be completed
    :param MID: array [n, v] containing on column i the availability info for kernel i; 1 means available, 0 missing
    :param approx_lvl: Nyström approximation level for getting the features (percentage, 0.1 means 10% of full)
    :param rank_lvl: the rank level of U that is learned (percentage, 0.1 means 10% of full)
    :param normalize: if the result is normalized. CVKT SHOULD ALWAYS USE NORMALIZATION IN POST-PROCESSING! REMOVING
                      NORMALIZATION WILL MAKE CVKT RESULTS UNCOMPARABLE TO OTHER ALGORITHMS
    :return: numpy array of size [n, n, v] containing the predicted (completed) kernel matrices
    """

    Kc = np.zeros(K.shape)

    for vv in range(K.shape[2]):

        if len(np.where(MID[:, vv] == 0)[0]) > 0:  # if there are samples to be completed for this view

            Kc[:, :, vv] = cvkt_complete_one_view(K, MID, vv, approx_lvl, rank_lvl, normalize=normalize)

        else:

            Kc[:, :, vv] = K[:, :, vv]

    return Kc


# ================================================= error measures =====================================================

def completion_accuracy_one_view(orig, pred):

    return 1 - np.trace(np.dot(orig, pred))/(np.linalg.norm(orig, ord='fro')*np.linalg.norm(pred, ord='fro'))


def average_relative_error_one_view(orig, pred, availability):

    """
    :param orig: n*n matrix
    :param pred: n*n matrix
    :param availability: list of length n containing 0 if data sample is missing, 1 otherwise
    :return: average relative error as presented in Bhadra et al
    """

    # predicted values of row - true values of row, 2 norm
    # divided by true values of row, 2 norm
    # sum over the rows that have are not observed
    # divide with n_missing

    rows = np.argwhere(availability == 0)
    res = 0
    for row in rows:
        res += np.linalg.norm(pred[row, :] - orig[row, :], ord=2)/np.linalg.norm(orig[row, :], ord=2)
    res = res / (orig.shape[0]-sum(availability))

    return res


