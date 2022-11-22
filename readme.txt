UPDATE 11/2022:
The initial CVKT code was developed using the Pymanopt version availavle at that time. With the current stable release, its syntax has changed majorly. The changes in CVKT reflect the current Pymanopt syntax. The old version remains as "cvkt_helpers_old.py". 


=======================================
= Code for Cross-View Kernel Transfer =
=======================================

Author: Riikka Huusari

Based on publication: Kernel transfer over multiple views for missing data completion by Riikka Huusari, Ćecile Capponi, Paul Villoutreix and Hachem Kadri


This code publication contains 1) code implementing the CVKT algorithm, and 2) datasets used in the above publication.

Main dependencies: Python 3, pymanopt (https://pymanopt.github.io/)

  - small bugs have been fixed 01/2020  (they had only been present in this tidied version for publication)

************
*** Code ***
************

* cvkt.py
  - The main file containing the function to call for CVKT.
  - Functions implementing the error measures CA and ARE.

* cvkt_helpers.py
  - Contains helper functions needed to run CVKT.


************
*** Data ***
************

Data used in the publication is provided in folders corresponding to the dataset. The data is saved in .mat files in order to facilitate the usage of competing MKC code, as it is easy to load these files in Python, too. 


* digits consists of the following files:
  - Kernel matrices in (200, 200, 6) array in "digits_K_n200.mat" as variable "K"
  - Labels in "digits_labels_n200.mat" as variable "y"
  - Missing value identifiers, (200, 6)-sized {0,1}-valued matrices, for various levels of missing data in files "digits_MID_missingX_n200.mat" as variable "MID". Here X is in [10, 20, 30, 40, 50, 60], indicating percentage of missing data to be completed.
 

* embryo
  - Files "viewX_kernel.mat" with X in [1, ..., 5] contain full kernel matrices calculated with available data, in variable named "view_kernel". Note that due to the missing values in data, the kernel matrices are of various sizes.
  - Folder "more_missing" contains the data for experimental configuration detailed in the paper
    - "embryoPartialExp_viewX_missingY_K.mat", X in [0, ..., 4] (note conflicting numbering in this subfolder to the previous; MATLAB numbering vs Python.) and Y in [10, 20, 30] (only results with 30 shown in the paper).
    - "embryoPartialExp_newKernels_viewX_missingY_MID.mat", X in [0, ..., 4] and Y in [10, 20, 30].
    - Here K contains the view to be completed (the one specified in the filename) first, then other views that are relevant to the problem after. MID follows this convention, adding some missing data to the first view. 


* simulated
  - File "sim_rbf_K.mat" contains a (100, 100, 7) array in variable "K".
  - Files "sim_rbf_missingX_MID.mat" contain the missing indices in "MID" as (100, 7) matrices, with X in [1, 2, 3, 4] the parameter a in the paper: number of views missing for each sample.



