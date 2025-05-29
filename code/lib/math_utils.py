#%% Import libraries

import math
import random

import numpy as np
import itertools as it

from scipy import stats
from skbio import DistanceMatrix
from skbio.stats import distance
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, euclidean

#%% Define functions for general purpose

def normalize_vec(vector):
    
    norm = np.linalg.norm(vector)
    if norm == 0: 
        
       return vector
   
    return vector/norm


def upper_triangular_to_flat(matrix):
    
    row_count = matrix.shape[0]
    indices = np.arange(row_count)
    mask = indices[:, None] < indices
    
    return matrix[mask]


def calc_geometric_median(input_vecs, eps = 1e-5):
    
    # Based on Yehuda Vardi and Cun-Hui Zhang's algorithm (2004)

    median = np.mean(input_vecs, 0)

    while True:
        d = cdist(input_vecs, [median])
        nonzeros = (d != 0)[:, 0]

        d_inv = 1 / d[nonzeros]
        d_invs = np.sum(d_inv)
        w = d_inv / d_invs
        t = np.sum(w * input_vecs[nonzeros], 0)
        num_zeros = len(input_vecs) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = t
            
        elif num_zeros == len(input_vecs):
            
            return median
        
        else:
            r = (t - median) * d_invs
            r = np.linalg.norm(r)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * t + min(1, rinv) * median

        if euclidean(median, y1) < eps:
            
            return y1

        median = y1


#%% Define functions for statistical tests

def calc_anova_for_means(groups):
    
    # ANOVA test of means
    
    k = len(groups)
    n = 0
    all_mean = 0
    means = []

    for group in groups:
        n += group.size
        mean = group.mean()
        means.append(mean)
        all_mean += mean * group.size
    
    all_mean /= n
  
    df1 = k - 1
    df2 = n - k
    
    if (df1 > 0 and df2 > 0):
        var_within = 0.0
        var_between = 0.0
        for i in range(k):
            var_between += np.square(means[i] - all_mean) * groups[i].size
            var_within += np.sum([np.square(e - means[i]) for e in groups[i]])
        if (var_within != 0):
            f_score = (var_between / var_within) * (df2 / df1)
            p_val = calc_fdist_pvalue(df1, df2, f_score)
            
            return f_score, p_val
    
    return 'nan', 'nan'


def calc_welch_anova_for_means(groups):
    
    # Welch ANOVA test of means
    
    error = False
    k = len(groups)
    
    for i in range(k):
        mean = groups[i].mean()
        if (groups[i].size < 2 or (np.sum([np.square(e - mean) for e in groups[i]])) == 0):
            error = True
    if not error:
        n = 0
        all_weight = 0    
        all_mean = 0            
        means = []
        weights = []
        for i in range(k):
            n += groups[i].size
            mean = groups[i].mean()
            means.append(mean)
            
            weight = groups[i].size / (np.sum([np.square(e - mean) for e in groups[i]]))
            weights.append(weight)
            
            all_weight += weight
            all_mean += weight * mean
                
        all_mean = all_mean / all_weight
        df2_denom = 0
         
        for i in range(k):
            df2_denom += (1 / (groups[i].size - 1)) * np.square(1 - (weights[i] / all_weight))
          
        df1 = k - 1
        df2 = (np.square(k) - 1) / (3 * df2_denom)
        if (df1 > 0 and df2 > 0):
            numer = 0.0
            denomer = 0.0
            for i in range(k):
                numer += np.square(means[i] - all_mean) * weights[i]
                denomer += (1 / (groups[i].size - 1)) * np.square(1 - (weights[i] / all_weight))
                
            f_score = (1 / (k - 1)) * numer / (1 + ((2 * (k - 2)) / (np.square(k) - 1)) * denomer)
            p_val = calc_fdist_pvalue(df1, df2, f_score)
            
            return f_score, p_val

    return 'nan', 'nan'


def calc_brown_for_means(groups):
    
    # Brown-Forsythe test of means
    
    error = False
    k = len(groups)
    
    for i in range(k):
        if (groups[i].size < 2):
            error = True
    if not error:
        n = 0
        all_mean = 0
        means = []
        for group in groups:
            n += group.size
            mean = group.mean()
            means.append(mean)
            all_mean += mean * group.size
        
        all_mean /= n
        df1 = k - 1
        
        if (df1 > 0):
            var_within = 0.0
            var_between = 0.0
            df2_recip = 0.0
            for i in range(k):
                var_between += np.square(means[i] - all_mean) * groups[i].size
                var_within += (np.sum([np.square(e - means[i]) for e in groups[i]])) * (1 - (groups[i].size / n))
            if (var_within > 0):
                for i in range(k):
                    df2_recip += (np.square((1 - ( groups[i].size / n)) * (np.sum([np.square(e - means[i]) for e in groups[i]]))
                                 / var_within) / (groups[i].size - 1))
                if (df2_recip > 0):
                    df2 = 1 / df2_recip
                    f_score = var_between / var_within
                    p_val = calc_fdist_pvalue(df1, df2, f_score)
                    
                    return f_score, p_val

    return 'nan', 'nan'


def calc_mannwhitney_for_means(groups):
    
    # Mann-Whitney test of means
    
    error = False
    k = len(groups)
    
    for i in range(k):
        if (groups[i].size < 2 or np.all(groups[i] == groups[i][0])):
            error = True
    if not error:
        f_score, p_val = stats.mannwhitneyu(groups[0], groups[1])

        return f_score, p_val

    return 'nan', 'nan'


def calc_f_for_vars(groups):
    
    # F-ratio test of variances
              
    means = []

    for group in groups:
        mean = group.mean()
        means.append(mean)
      
    df1 = groups[0].size - 1
    df2 = groups[1].size - 1
    
    if (df1 > 0 and df2 > 0):
        var_1 = (np.sum([np.square(e - means[0]) for e in groups[0]])) * (1 / df1)
        var_2 = (np.sum([np.square(e - means[1]) for e in groups[1]])) * (1 / df2)
        if (var_1 >= var_2 and var_2 > 0):
            f_score = var_1 / var_2
            p_val = calc_incomplete_beta(0.5 * df1, 0.5 * df2, 
                                         float(df1) * f_score/(df1 * f_score + df2))
            return f_score, p_val
            
        elif (var_1 < var_2 and var_1 > 0):
            f_score = var_2 / var_1
            p_val = calc_fdist_pvalue(df1, df2, f_score)
            
            return f_score, p_val
    
    return 'nan', 'nan'


def calc_brown_or_levene_for_vars(groups, mode):
    
    # Brown-Forsythe or Levene test of variances
    
    k = len(groups)
    n = 0
    medians = []

    for group in groups:
        if mode == 'brown':
            median = np.nanmedian(group, overwrite_input = True)
        elif mode == 'levene':
            median = np.nanmean(group)
            
        n += group.size    
        medians.append(median)
        
    df1 = k - 1
    df2 = n - k
    
    if (df1 > 0 and df2 > 0):
        groups_transform = []
        for i in range(k):
            group_transform = np.zeros((groups[i].size))
            for j in range(groups[i].size):
                group_transform[j] = np.abs(groups[i][j] - medians[i])
                
            groups_transform.append(group_transform)
        
        means = []
        all_mean = 0
        
        for group_transform in groups_transform:
            mean = group_transform.mean()
            means.append(mean)
            all_mean += mean * group_transform.size
            
        all_mean = all_mean / n
        var_within = 0.0
        var_between = 0.0
        
        for i in range(k):
            var_between += np.square(means[i] - all_mean) * groups_transform[i].size
            var_within += np.sum([np.square(e - means[i]) for e in groups_transform[i]])
        if (var_within != 0): 
            f_score = (var_between / var_within) * (df2 / df1)
            p_val = calc_fdist_pvalue(df1, df2, f_score)
            
            return f_score, p_val
    
    return 'nan', 'nan'


def calc_bartlett_for_vars(groups):
    
    # Bartlett test for variances
    
    error = False
    k = len(groups)
    
    for i in range(k):
        if (groups[i].size < 2 or np.var(groups[i], ddof = 1) == 0):
            error = True
    if not error:
        f_score, p_val = stats.bartlett(*groups)

        return f_score, p_val

    return 'nan', 'nan'


def perform_permdisp_pairwise_dist(distmat, group_indices, perm_num):
    
    np_matrix_adjusted = adjust_matrix_for_centered_permutation(distmat, group_indices)
    fscores = []
    fscores.append((calc_permdisp_pairwise_dist(np_matrix_adjusted, group_indices, False))[0])
    perm_range = list(range(len(group_indices[0]) + len(group_indices[1])))
    index_permuations = []
    
    if (math.factorial(len(perm_range)) > perm_num):
        while len(index_permuations) < perm_num:
            temp = perm_range.copy()
            random.shuffle(temp)
            index_permuations.append(temp)
    else: 
        perm_list = list(it.permutations(perm_range))
        index_permuations = random.sample(perm_list, len(perm_list))
    
    for perm in index_permuations:
                                            
        np_matrix_recombined = np.take(np_matrix_adjusted, perm, 0)
        np_matrix_recombined = np.take(np_matrix_recombined, perm, 1)
        
        fscore = calc_permdisp_pairwise_dist(np_matrix_recombined, group_indices, False)[0]
        fscores.append(fscore)

    sorted_fscores = stats.rankdata(fscores, method = 'max')
    p_val = (len(index_permuations) - int(sorted_fscores[0]) + 2) / (len(index_permuations) + 1)
    
    return fscores[0], p_val

                        
def calc_permdisp_pairwise_dist(distmat, groups, calc_p):
    
    # Test for homogenity of multivariate dispersions, using Cailliez correlation
    # permutation method by Gijbels et al. (2012)
    
    error = False
    
    for group in groups:
        if (len(group) < 3):
            error = True
    if not error:
        
        k = len(groups)
        n = 0
        nks = []
        dks = []
        
        sigma_sq = 0
        d = 0
        
        for group in groups:
            
            dk_matrix = np.take(distmat, group, 0)
            dk_matrix = np.take(dk_matrix, group, 1)
            dk_vec = upper_triangular_to_flat(dk_matrix)
            dk_vals = dk_vec[dk_vec > -np.inf]
            dk = dk_vals.mean()
            dks.append(dk)
            
            nk = len(group)
            nks.append(nk)
            
            n += nk
            d += nk * dk
            
            s_sq = 0
            
            for i in range(len(group)):
                
                di_vec = np.take(dk_matrix, i, 0)
                di_pre = di_vec[di_vec > -np.inf]
                di_vals = np.delete(di_pre, i)
                s_sq += (di_vals.mean() - dk) ** 2
                
            s_sq = ((4 * (nk - 1)) / ((nk - 2) ** 2)) * s_sq
            sigma_sq += (nk - 1) * s_sq
        
        df1 = k - 1
        df2 = n - k
        
        if (df1 > 0 and df2 > 0):
            
            sigma_sq = sigma_sq / (n - k)
            d = d / n
            
            nominator = 0
            
            for i in range(len(groups)):
                        
                nominator += nks[i] * ((dks[i] - d) ** 2)
        
            f_score = nominator / ((k - 1) * (sigma_sq ** 2))
            
            if calc_p:
                return f_score, calc_fdist_pvalue(df1, df2, f_score)
            else:
                return f_score, 'nan'

    return 'nan', 'nan'


def calc_permanova_for_means(distmat, groups, permutations):
        
    error = False
    ids = []
    grouping = []
    
    for i in range(len(groups)):
        if (len(groups[i]) < 3):
            error = True
            break
        
        ids.extend(groups[i])
        grouping.extend([str(i)] * len(groups[i]))
    
    np.fill_diagonal(distmat, 0)
    if not error and - np.inf not in distmat:
        dm = DistanceMatrix(distmat, ids)
        result = distance.permanova(dm, grouping, None, permutations)
        
        return result['test statistic'], result['p-value']

    return 'nan', 'nan'


#%% Define functions for p-value calculation

def calc_fdist_pvalue(df1, df2, f_score):
    
    return calc_incomplete_beta(0.5 * df1, 0.5 * df2, 
                                float(df1) * f_score/(df1 * f_score + df2))

def calc_incomplete_beta(a, b, x):

    # Evaluates incomplete beta function, here a, b > 0 and 0 <= x <= 1.
    # This function requires contfractbeta(a, b, x, itmax = 200)

    if (x == 0):
        return 1
    elif (x == 1):                
        return 0
    else:
        lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) 
                 + a * math.log(x) + b * math.log(1 - x))
        
        if (x < (a + 1) / (a + b + 2)):
            fract = calc_cont_fractbeta(a, b, x) / a
            if fract != 'nan':
                return 1 - (math.exp(lbeta) * fract)
            else:
                return 'nan'
        
        else:
            fract = calc_cont_fractbeta(b, a, 1 - x)
            if fract != 'nan':
                return 1 - (1 - math.exp(lbeta) * (fract / b))
            
            else:
                return 'nan'
            

def calc_cont_fractbeta(a, b, x, itmax = 800):

    # Evaluates the continued fraction form of the incomplete Beta function

    eps = 3.0e-7
    bm = az = am = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap

    for i in range(itmax + 1):
        
        em = float(i + 1)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((qap + tem) * (a + tem))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if (abs(az - aold) < (eps * abs(az))):
            return az

    # print('a or b too large or given itmax too small for computing incomplete beta function.')
    return 'nan'
    

#%% Define functions for ordination

def adjust_matrix_for_centered_permutation(distmat, groups):
    
    n = distmat.shape[0]
    eigs_c_matrix, matrix_A_hat_centred = calc_corrected_matrix_by_cailliez_method(distmat)
    dk_matrix = calc_pcoa_representation(matrix_A_hat_centred, n)
    
    for group in groups:
        
        # tk = np.mean(dk_matrix[group, :], axis = 0)
        tk = calc_geometric_median(dk_matrix[group, :])
        dk_matrix[group, :] = [x - tk for x in dk_matrix[group, :]]
    
    matrix_final = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix_final[i, j] = np.linalg.norm(dk_matrix[i, :] - dk_matrix[j, :])
                
    return matrix_final - eigs_c_matrix


def calc_pcoa_representation(distmat, components):
    
    eigvals, eigvecs = eigh(distmat)
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]
    num_positive = (eigvals >= 0).sum()
    eigvecs[:, num_positive:] = np.zeros(eigvecs[:, num_positive:].shape)
    eigvals[num_positive:] = np.zeros(eigvals[num_positive:].shape)
    dk_matrix = eigvecs[0:components, :] * np.sqrt(eigvals[0:components])
    
    return dk_matrix
    
    
def calc_corrected_matrix_by_cailliez_method(distmat):
    
    n = distmat.shape[0]
    np.fill_diagonal(distmat, 0)
    matrix_A = np.zeros((2 * n, 2 * n))
    matrix_I = np.zeros((n, n))
    np.fill_diagonal(matrix_I, -1)
    matrix_a1 = -0.5 * (distmat ** 2)
    matrix_a2 = -0.5 * distmat
    a1 = matrix_a1.mean()
    a2 = matrix_a2.mean()
    
    for i in range(n):
        ai1 = matrix_a1[i, :].mean()
        ai2 = matrix_a2[i, :].mean()
        for j in range(n):
            aj1 = matrix_a1[:, j].mean()
            aj2 = matrix_a2[:, j].mean()
            
            matrix_A[i, n + j] = 2 * (matrix_a1[i, j] - ai1 - aj1 + a1)
            matrix_A[n + i, j] = matrix_I[i, j]
            matrix_A[n + i, n + j] = -4 * (matrix_a2[i, j] - ai2 - aj2 + a2)
            
    eigs_cs = eigh(matrix_A, eigvals_only = True)
    idxs_descending = eigs_cs.argsort()[::-1]
    eigs_cs = eigs_cs[idxs_descending]
    eigs_c_matrix = np.zeros((n, n)) + eigs_cs[0]
    np.fill_diagonal(eigs_c_matrix, 0)
    distmat = distmat + eigs_c_matrix
    
    matrix_A_hat =  -0.5 * (distmat ** 2)
    matrix_A_hat_centred = np.zeros((n, n))
    
    a = matrix_A_hat.mean()
    
    for i in range(n):
        ai = matrix_A_hat[i, :].mean()
        for j in range(n):
            aj = matrix_A_hat[:, j].mean()
            
            matrix_A_hat_centred[i, j] = matrix_A_hat[i, j] - ai - aj + a
            
    return eigs_c_matrix, matrix_A_hat_centred
