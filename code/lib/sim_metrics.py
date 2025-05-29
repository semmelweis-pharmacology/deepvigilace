#%% Import libraries

import math

import numpy as np

from scipy.spatial import distance

#%% Define functions for vector-based similarity measurements

def calculate_cosine_single(data_dict, center_point, data_indices, target_index):
        
    sim = 1 - distance.cosine(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_cosine_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('cosine progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = 1 - distance.cosine(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_euclidean_single(data_dict, center_point, data_indices, target_index):
        
    sim = -np.linalg.norm(data_dict[center_point] - data_indices[target_index])

    return sim


def calculate_euclidean_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('euclidean progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -np.linalg.norm(data_indices[center_point_index] - value)
        sim_dict[key] = sim

    return sim_dict


def calculate_dot_single(data_dict, center_point, data_indices, target_index):
        
    sim = np.dot(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_dot_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('dot progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = np.dot(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_manhattan_single(data_dict, center_point, data_indices, target_index):
        
    sim = -distance.cityblock(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_manhattan_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('manhattan progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.cityblock(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_braycurtis_single(data_dict, center_point, data_indices, target_index):
        
    sim = -distance.braycurtis(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_braycurtis_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('braycurtis progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.braycurtis(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_canberra_single(data_dict, center_point, data_indices, target_index):
        
    sim = -distance.canberra(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_canberra_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('canberra progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.canberra(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_chebyshev_single(data_dict, center_point, data_indices, target_index):
        
    sim = -distance.chebyshev(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_chebyshev_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('chebyshev progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.chebyshev(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_correlation_single(data_dict, center_point, data_indices, target_index):
        
    sim = 1 - distance.correlation(data_dict[center_point], data_indices[target_index])

    return sim

def calculate_correlation_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('correlation progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = 1 - distance.correlation(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_minkowski_single(data_dict, center_point, data_indices, target_index):
        
    sim = -distance.minkowski(data_dict[center_point], data_indices[target_index], p = 1.3)

    return sim


def calculate_minkowski_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('minkowski progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.minkowski(data_indices[center_point_index], value, p = 1.3)
        sim_dict[key] = sim

    return sim_dict


def calculate_seuclidean_single(data_dict, center_point, data_indices, var, target_index):
        
    sim = -distance.seuclidean(data_dict[center_point], data_indices[target_index], var)

    return sim


def calculate_seuclidean_for_all(data_dict, data_indices, var, center_point_index):
    
    sim_dict = {}
    # print('seuclidean progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.seuclidean(data_indices[center_point_index], value, var)
        sim_dict[key] = sim

    return sim_dict


def calculate_sqeuclidean_single(data_dict, center_point, data_indices, target_index):
        
    sim = -distance.sqeuclidean(data_dict[center_point], data_indices[target_index])

    return sim


def calculate_sqeuclidean_for_all(data_dict, data_indices, center_point_index):
    
    sim_dict = {}
    # print('sqeuclidean progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        sim = -distance.sqeuclidean(data_indices[center_point_index], value)
        sim_dict[key] = sim

    return sim_dict


def calculate_mahalanobis_single(data_dict, center_point, data_indices, mean, inv_cov, target_index):
    
    # sim_base = np.absolute((distance.mahalanobis(data_dict[center_point], mean, inv_cov) - distance.mahalanobis(data_indices[target_index], mean, inv_cov)))
    sim_pair = distance.mahalanobis(data_dict[center_point], data_indices[target_index], inv_cov)

    # return - sim_base - sim_pair
    return -sim_pair


def calculate_mahalanobis_for_all(data_dict, data_indices, mean, inv_cov, center_point_index):
    
    sim_dict = {}
    # print('mahalanobis progress: ' + str(center_point_index) + '/' + str(len(data_dict)))
    
    for key, value in data_dict.items():
        
        # sim_base = np.absolute((distance.mahalanobis(data_indices[center_point_index], mean, inv_cov) - distance.mahalanobis(value, mean, inv_cov)))
        sim_pair = (distance.mahalanobis(data_indices[center_point_index], value, inv_cov))
        sim_dict[key] = - sim_pair

    return sim_dict


#%% Define functions for hierarchy-based similarity measurements

def calculate_meanminpath_single(data_dict, center_point, data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in data_dict):
        for hierarchy_center in data_dict[center_point]:
            for hierarchy_target in data_dict[data_keys[target_index]]:
                if hierarchy_center['HLT'] == hierarchy_target['HLT']:              
                    minpath = 2                   
                elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:                    
                    minpath = 4                    
                elif hierarchy_center['SOC'] == hierarchy_target['SOC']:                   
                    minpath = 6                   
                else:                 
                    minpath = 8
                    
                sim = sim + 1/minpath
                
        sim = sim/(len(data_dict[center_point]) * len(data_dict[data_keys[target_index]]))     
    else:
        sim = -np.inf
    
    return sim

def calculate_meanminpath_for_all(data_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('meanminpath progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in data_dict:     
        for target_key in data_keys:         
            sim = 0   
            if target_key in data_dict:                       
                for hierarchy_center in data_dict[data_keys[center_point_index]]:              
                    for hierarchy_target in data_dict[target_key]:             
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:                  
                            minpath = 2                 
                        elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:                     
                            minpath = 4                     
                        elif hierarchy_center['SOC'] == hierarchy_target['SOC']:                    
                            minpath = 6                   
                        else:                   
                            minpath = 8
                            
                        sim = sim + 1/minpath
                        
                sim = sim/(len(data_dict[data_keys[center_point_index]]) * len(data_dict[target_key]))         
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim    
    else:
        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_primaryminpath_single(data_dict, center_point, data_keys, target_index):

    if (data_keys[target_index] in data_dict):              
        for hierarchy_center in data_dict[center_point]:      
            if hierarchy_center['RANK'] == 'PRIMARY':     
                for hierarchy_target in data_dict[data_keys[target_index]]:            
                    if hierarchy_target['RANK'] == 'PRIMARY':        
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:                   
                            minpath = 2                  
                        elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:                  
                            minpath = 4                
                        elif hierarchy_center['SOC'] == hierarchy_target['SOC']:                          
                            minpath = 6                          
                        else:                          
                            minpath = 8
                            
        sim = 1/minpath                       
    else:
        sim = -np.inf
                    
    return sim


def calculate_primaryminpath_for_all(data_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('primaryminpath progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in data_dict:      
        for target_key in data_keys:         
            sim = 0    
            if target_key in data_dict:                        
                for hierarchy_center in data_dict[data_keys[center_point_index]]:              
                    if hierarchy_center['RANK'] == 'PRIMARY':             
                        for hierarchy_target in data_dict[target_key]:            
                            if hierarchy_target['RANK'] == 'PRIMARY':                        
                                if hierarchy_center['HLT'] == hierarchy_target['HLT']:                               
                                    minpath = 2                           
                                elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:                             
                                    minpath = 4                               
                                elif hierarchy_center['SOC'] == hierarchy_target['SOC']:                               
                                    minpath = 6                              
                                else:                             
                                    minpath = 8
                                                        
                sim = 1/minpath            
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim       
    else:

        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_meanpath_single(data_dict, center_point, data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in data_dict):
        for hierarchy_center in data_dict[center_point]:     
            for hierarchy_target in data_dict[data_keys[target_index]]:         
                if hierarchy_center['HLT'] == hierarchy_target['HLT']:              
                    sim = sim + 1/2             
                if hierarchy_center['HLGT'] == hierarchy_target['HLGT']:            
                    sim = sim + 1/4           
                if hierarchy_center['SOC'] == hierarchy_target['SOC']:  
                    sim = sim + 1/6                
                if (hierarchy_center['HLT'] != hierarchy_target['HLT']
                    and hierarchy_center['HLGT'] != hierarchy_target['HLGT']
                    and hierarchy_center['SOC'] != hierarchy_target['SOC']):
                    
                    sim = sim + 1/8
                
        sim = sim/(len(data_dict[center_point]) * len(data_dict[data_keys[target_index]]))     
    else:
        sim = -np.inf
    
    return sim


def calculate_meanpath_for_all(data_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('meanpath progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in data_dict:   
        for target_key in data_keys:       
            sim = 0   
            if target_key in data_dict:                      
                for hierarchy_center in data_dict[data_keys[center_point_index]]:             
                    for hierarchy_target in data_dict[target_key]:                 
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:                     
                            sim = sim + 1/2                    
                        elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:                   
                            sim = sim + 1/4                  
                        elif hierarchy_center['SOC'] == hierarchy_target['SOC']:                      
                            sim = sim + 1/6                       
                        else:                       
                            sim = sim + 1/8
                                                    
                sim = sim/(len(data_dict[data_keys[center_point_index]]) * len(data_dict[target_key]))       
            else:
                sim = -np.inf
           
            sim_dict[target_key] = sim  
    else:

        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_primarypath_single(data_dict, center_point, data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in data_dict):  
        for hierarchy_center in data_dict[center_point]:  
            if hierarchy_center['RANK'] == 'PRIMARY': 
                for hierarchy_target in data_dict[data_keys[target_index]]:       
                    if hierarchy_target['RANK'] == 'PRIMARY':  
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:             
                            sim = sim + 1/2             
                        if hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                            sim = sim + 1/4
                        if hierarchy_center['SOC'] == hierarchy_target['SOC']:
                            sim = sim + 1/6
                        if (hierarchy_center['HLT'] != hierarchy_target['HLT']
                            and hierarchy_center['HLGT'] != hierarchy_target['HLGT']
                            and hierarchy_center['SOC'] != hierarchy_target['SOC']):
                            
                            sim = sim + 1/8               
    else:
        sim = -np.inf
                    
    return sim


def calculate_primarypath_for_all(data_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('primarypath progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in data_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in data_dict:
                for hierarchy_center in data_dict[data_keys[center_point_index]]:
                    if hierarchy_center['RANK'] == 'PRIMARY':
                        for hierarchy_target in data_dict[target_key]:
                            if hierarchy_target['RANK'] == 'PRIMARY':
                                if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                                    sim = sim + 1/2
                                elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                                    sim = sim + 1/4
                                elif hierarchy_center['SOC'] == hierarchy_target['SOC']:
                                    sim = sim + 1/6
                                else:
                                    sim = sim + 1/8
                                                                        
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:

        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_smqmeanminpath_single(data_dict, center_point, data_keys, target_index):
        
    sim = 0
    
    if data_keys[target_index] in data_dict:
        for hierarchy_center in data_dict[center_point]:
            for hierarchy_target in data_dict[data_keys[target_index]]:
                if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                and hierarchy_center['SMQ5'] != '-'):
                    minpath = 2
                elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                and hierarchy_center['SMQ4'] != '-'):
                    minpath = 4
                elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                and hierarchy_center['SMQ3'] != '-'):
                    minpath = 6
                elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                and hierarchy_center['SMQ2'] != '-'):
                    minpath = 8
                elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                and hierarchy_center['SMQ1'] != '-'):
                    minpath = 10
                else:
                    minpath = 12
                    
                sim = sim + 1/minpath
                
        sim = sim/(len(data_dict[center_point]) * len(data_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_smqmeanminpath_for_all(data_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('smqmeanminpath progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in data_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in data_dict:
                for hierarchy_center in data_dict[data_keys[center_point_index]]:
                    for hierarchy_target in data_dict[target_key]:
                        if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                        and hierarchy_center['SMQ5'] != '-'):
                            minpath = 2
                        elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                        and hierarchy_center['SMQ4'] != '-'):
                            minpath = 4
                        elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                        and hierarchy_center['SMQ3'] != '-'):
                            minpath = 6
                        elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                        and hierarchy_center['SMQ2'] != '-'):
                            minpath = 8
                        elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                        and hierarchy_center['SMQ1'] != '-'):
                            minpath = 10
                        else:
                            minpath = 12
                            
                        sim = sim + 1/minpath
                                                    
                sim = sim/(len(data_dict[data_keys[center_point_index]]) * len(data_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:

        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


#%% Define functions for information-content-based similarity measurements

def calculate_resnikmeddra_single(ic_dict, hier_dict, center_point, data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in hier_dict):
        for hierarchy_center in hier_dict[center_point]:
            for hierarchy_target in hier_dict[data_keys[target_index]]:
                if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                    sim = sim + ic_dict[hierarchy_center['HLT']]
                elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                    sim = sim + ic_dict[hierarchy_center['HLGT']]
                elif hierarchy_center['SOC'] == hierarchy_target['SOC']:
                    sim = sim + ic_dict[hierarchy_center['SOC']]
                elif (hierarchy_center['HLT'] != hierarchy_target['HLT']
                    and hierarchy_center['HLGT'] != hierarchy_target['HLGT']
                    and hierarchy_center['SOC'] != hierarchy_target['SOC']):
                    
                    sim = sim + -math.log(1, 2)
                
        sim = sim/(len(hier_dict[center_point]) * len(hier_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_resnikmeddra_for_all(ic_dict, hier_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('resnikmeddra progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in hier_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in hier_dict:
                for hierarchy_center in hier_dict[data_keys[center_point_index]]:
                    for hierarchy_target in hier_dict[target_key]:
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                            sim = sim + ic_dict[hierarchy_center['HLT']]
                        elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                            sim = sim + ic_dict[hierarchy_center['HLGT']]
                        elif hierarchy_center['SOC'] == hierarchy_target['SOC']:
                            sim = sim + ic_dict[hierarchy_center['SOC']]
                        else:
                            sim = sim + (-math.log(1, 2))
                                                    
                sim = sim/(len(hier_dict[data_keys[center_point_index]]) * len(hier_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:
        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_resniksmq_single(ic_dict, hier_dict, center_point, data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in hier_dict):
        for hierarchy_center in hier_dict[center_point]:
            for hierarchy_target in hier_dict[data_keys[target_index]]:
                if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                and hierarchy_center['SMQ5'] != '-'):
                     sim = sim + ic_dict[hierarchy_center['SMQ5']]
                elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                and hierarchy_center['SMQ4'] != '-'):
                    sim = sim + ic_dict[hierarchy_center['SMQ4']]
                elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                and hierarchy_center['SMQ3'] != '-'):
                    sim = sim + ic_dict[hierarchy_center['SMQ3']]
                elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                and hierarchy_center['SMQ2'] != '-'):
                    sim = sim + ic_dict[hierarchy_center['SMQ2']]
                elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                and hierarchy_center['SMQ1'] != '-'):
                    sim = sim + ic_dict[hierarchy_center['SMQ1']]
                else:
                    sim = sim + -math.log(1, 2)
                
        sim = sim/(len(hier_dict[center_point]) * len(hier_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_resniksmq_for_all(ic_dict, hier_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('resniksmq progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in hier_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in hier_dict:
                for hierarchy_center in hier_dict[data_keys[center_point_index]]:                    
                    for hierarchy_target in hier_dict[target_key]:
                        if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                        and hierarchy_center['SMQ5'] != '-'):
                             sim = sim + ic_dict[hierarchy_center['SMQ5']]
                        elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                        and hierarchy_center['SMQ4'] != '-'):
                            sim = sim + ic_dict[hierarchy_center['SMQ4']]
                        elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                        and hierarchy_center['SMQ3'] != '-'):
                            sim = sim + ic_dict[hierarchy_center['SMQ3']]
                        elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                        and hierarchy_center['SMQ2'] != '-'):
                            sim = sim + ic_dict[hierarchy_center['SMQ2']]
                        elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                        and hierarchy_center['SMQ1'] != '-'):
                            sim = sim + ic_dict[hierarchy_center['SMQ1']]
                        else:
                            sim = sim + -math.log(1, 2)
                                                    
                sim = sim/(len(hier_dict[data_keys[center_point_index]]) * len(hier_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:
        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_jiangconrathmeddra_single(ic_dict, hier_dict, center_point, 
                                        data_keys, target_index):
    
    sim = 0
    if (data_keys[target_index] in hier_dict and data_keys[target_index] != center_point):
        for hierarchy_center in hier_dict[center_point]:
            for hierarchy_target in hier_dict[data_keys[target_index]]:
                if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['HLT']])
                elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['HLGT']])
                elif hierarchy_center['SOC'] == hierarchy_target['SOC']:
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['SOC']])
                elif (hierarchy_center['HLT'] != hierarchy_target['HLT']
                    and hierarchy_center['HLGT'] != hierarchy_target['HLGT']
                    and hierarchy_center['SOC'] != hierarchy_target['SOC']):
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * (-math.log(1, 2)))
                
        sim = sim/(len(hier_dict[center_point]) * len(hier_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_jiangconrathmeddra_for_all(ic_dict, hier_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('jiangconrathmeddra progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in hier_dict:
        for target_key in data_keys:
            sim = 0
            if (target_key in hier_dict and data_keys[center_point_index] != target_key):                            
                for hierarchy_center in hier_dict[data_keys[center_point_index]]:
                    for hierarchy_target in hier_dict[target_key]:
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                            sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['HLT']])
                        elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                            sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['HLGT']])                       
                        elif hierarchy_center['SOC'] == hierarchy_target['SOC']:                            
                            sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['SOC']])                      
                        else:
                            sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * (-math.log(1, 2)))

                sim = sim/(len(hier_dict[data_keys[center_point_index]]) * len(hier_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:
        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_jiangconrathsmq_single(ic_dict, hier_dict, center_point, 
                                        data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in hier_dict and data_keys[target_index] != center_point):
        for hierarchy_center in hier_dict[center_point]:
            for hierarchy_target in hier_dict[data_keys[target_index]]:
                if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                and hierarchy_center['SMQ5'] != '-'):
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['SMQ5']])
                elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                and hierarchy_center['SMQ4'] != '-'):
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['SMQ4']])
                elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                and hierarchy_center['SMQ3'] != '-'):
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['SMQ3']])
                elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                and hierarchy_center['SMQ2'] != '-'):
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['SMQ2']])
                elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                and hierarchy_center['SMQ1'] != '-'):
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * ic_dict[hierarchy_center['SMQ1']])
                else:
                    sim = sim + 1 / (ic_dict[center_point] + ic_dict[data_keys[target_index]] - 2 * (-math.log(1, 2)))
                
        sim = sim/(len(hier_dict[center_point]) * len(hier_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_jiangconrathsmq_for_all(ic_dict, hier_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('jiangconrathsmq progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in hier_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in hier_dict and target_key != data_keys[center_point_index]:
                for hierarchy_center in hier_dict[data_keys[center_point_index]]:
                    for hierarchy_target in hier_dict[target_key]:
                        if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                        and hierarchy_center['SMQ5'] != '-'):
                            sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['SMQ5']])
                        elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                        and hierarchy_center['SMQ4'] != '-'):
                             sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['SMQ4']])
                        elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                        and hierarchy_center['SMQ3'] != '-'):
                             sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['SMQ3']])
                        elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                        and hierarchy_center['SMQ2'] != '-'):
                             sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['SMQ2']])                            
                        elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                        and hierarchy_center['SMQ1'] != '-'):
                             sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * ic_dict[hierarchy_center['SMQ1']])
                        else:
                            sim = sim + 1 / (ic_dict[data_keys[center_point_index]] + ic_dict[target_key] - 2 * (-math.log(1, 2)))
                                                    
                sim = sim/(len(hier_dict[data_keys[center_point_index]]) * len(hier_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:
        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_linmeddra_single(ic_dict, hier_dict, center_point, 
                                        data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in hier_dict):
        for hierarchy_center in hier_dict[center_point]:
            for hierarchy_target in hier_dict[data_keys[target_index]]:
                if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                    sim = sim + (2 * ic_dict[hierarchy_center['HLT']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:                    
                    sim = sim + (2 * ic_dict[hierarchy_center['HLGT']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])                    
                elif hierarchy_center['SOC'] == hierarchy_target['SOC']:
                    sim = sim + (2 * ic_dict[hierarchy_center['SOC']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                elif (hierarchy_center['HLT'] != hierarchy_target['HLT']
                    and hierarchy_center['HLGT'] != hierarchy_target['HLGT']
                    and hierarchy_center['SOC'] != hierarchy_target['SOC']):                    
                    sim = sim + (2 * (-math.log(1, 2))) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                
        sim = sim/(len(hier_dict[center_point]) * len(hier_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_linmeddra_for_all(ic_dict, hier_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('linmeddra progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in hier_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in hier_dict:
                for hierarchy_center in hier_dict[data_keys[center_point_index]]:
                    for hierarchy_target in hier_dict[target_key]:
                        if hierarchy_center['HLT'] == hierarchy_target['HLT']:
                            sim = sim + (2 * ic_dict[hierarchy_center['HLT']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        elif hierarchy_center['HLGT'] == hierarchy_target['HLGT']:
                            sim = sim + (2 * ic_dict[hierarchy_center['HLGT']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        elif hierarchy_center['SOC'] == hierarchy_target['SOC']:
                            sim = sim + (2 * ic_dict[hierarchy_center['SOC']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        else:
                            sim = sim + (2 * (-math.log(1, 2))) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                                                    
                sim = sim/(len(hier_dict[data_keys[center_point_index]]) * len(hier_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:
        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict


def calculate_linsmq_single(ic_dict, hier_dict, center_point, 
                                        data_keys, target_index):
    
    sim = 0
    
    if (data_keys[target_index] in hier_dict):
        for hierarchy_center in hier_dict[center_point]:
            for hierarchy_target in hier_dict[data_keys[target_index]]:
                if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                and hierarchy_center['SMQ5'] != '-'):
                    sim = sim + (2 * ic_dict[hierarchy_center['SMQ5']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                and hierarchy_center['SMQ4'] != '-'):
                    sim = sim + (2 * ic_dict[hierarchy_center['SMQ4']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                and hierarchy_center['SMQ3'] != '-'):
                    sim = sim + (2 * ic_dict[hierarchy_center['SMQ3']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                and hierarchy_center['SMQ2'] != '-'):
                    sim = sim + (2 * ic_dict[hierarchy_center['SMQ2']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                and hierarchy_center['SMQ1'] != '-'):
                    sim = sim + (2 * ic_dict[hierarchy_center['SMQ1']]) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                else:
                    sim = sim + (2 * (-math.log(1, 2))) / (ic_dict[center_point] + ic_dict[data_keys[target_index]])
                
        sim = sim/(len(hier_dict[center_point]) * len(hier_dict[data_keys[target_index]]))
    else:
        sim = -np.inf
    
    return sim


def calculate_linsmq_for_all(ic_dict, hier_dict, data_keys, center_point_index):
    
    sim_dict = {}
    # print('linsmq progress: ' + str(center_point_index) + '/' + str(len(data_keys)))
    
    if data_keys[center_point_index] in hier_dict:
        for target_key in data_keys:
            sim = 0
            if target_key in hier_dict:
                for hierarchy_center in hier_dict[data_keys[center_point_index]]:
                    for hierarchy_target in hier_dict[target_key]:
                        if (hierarchy_center['SMQ5'] == hierarchy_target['SMQ5']
                        and hierarchy_center['SMQ5'] != '-'):
                            sim = sim + (2 * ic_dict[hierarchy_center['SMQ5']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        elif (hierarchy_center['SMQ4'] == hierarchy_target['SMQ4']
                        and hierarchy_center['SMQ4'] != '-'):
                            sim = sim + (2 * ic_dict[hierarchy_center['SMQ4']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        elif (hierarchy_center['SMQ3'] == hierarchy_target['SMQ3']
                        and hierarchy_center['SMQ3'] != '-'):
                            sim = sim + (2 * ic_dict[hierarchy_center['SMQ3']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        elif (hierarchy_center['SMQ2'] == hierarchy_target['SMQ2']
                        and hierarchy_center['SMQ2'] != '-'):
                            sim = sim + (2 * ic_dict[hierarchy_center['SMQ2']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        elif (hierarchy_center['SMQ1'] == hierarchy_target['SMQ1']
                        and hierarchy_center['SMQ1'] != '-'):
                            sim = sim + (2 * ic_dict[hierarchy_center['SMQ1']]) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                        else:
                            sim = sim + (2 * (-math.log(1, 2))) / (ic_dict[target_key] + ic_dict[data_keys[center_point_index]])
                                                    
                sim = sim/(len(hier_dict[data_keys[center_point_index]]) * len(hier_dict[target_key]))
            else:
                sim = -np.inf
            
            sim_dict[target_key] = sim
    else:

        zip_iterator = zip(data_keys, np.array(np.ones(len(data_keys)) * -np.inf))
        sim_dict = dict(zip_iterator)
    
    return sim_dict

    