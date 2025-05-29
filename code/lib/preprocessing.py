#%% Import libraries

import sys
import random
import math
import pandas as pd

import numpy as np

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#%% Define functions for dict processing

def sort_dict(in_dict):
    
    keys = list(in_dict.keys())
    keys.sort()
    sorted_dict = {i: in_dict[i] for i in keys}
    
    return sorted_dict


def build_dict_with_subsamp_dist(data, freq_map):
    
    new_dict_indices = {}
    new_dict_subsamp = {}
    for i, (key, value) in enumerate(freq_map.items()):
        new_dict_indices[key] = i
        new_dict_subsamp[key] = (np.sqrt(value/(0.00001 * len(data))) + 1) * (0.00001 * len(data))/value
        
    return new_dict_indices, new_dict_subsamp


def build_dict_with_unigram_dist(freq_map):
    
    new_dict_indices = {}
    new_dict_unigram = {}
    
    freq_sum = sum([float(k)**(3/4) for k in freq_map.values()])
    freq_normalized = [(float(k)**(3/4))/freq_sum for k in freq_map.values()]
    
    for i, (key, value) in enumerate(freq_map.items()):
        new_dict_indices[key] = i
        new_dict_unigram[key] = freq_normalized[i]
        
    return new_dict_indices, new_dict_unigram


def generate_sampling_maps(data):
    
    samp_map_drug = {}
    samp_map_reac = {}
    
    for index, row in data.iterrows():
        if row['DRUG'] not in samp_map_drug:
            samp_map_drug[row['DRUG']] = 1
        else:
            samp_map_drug[row['DRUG']] += 1
            
        if row['REACTION'] not in samp_map_reac:
            samp_map_reac[row['REACTION']] = 1
        else:
            samp_map_reac[row['REACTION']] += 1
            
    _, new_samp_map_drug = build_dict_with_unigram_dist(samp_map_drug)
    _, new_samp_map_reac = build_dict_with_unigram_dist(samp_map_reac)
    
    return new_samp_map_drug, new_samp_map_reac


def process_freq_for_ic_pt(data_dict):

    ic_dict = {}
    freq_sum = sum(list(data_dict.values()))

    for key, value in data_dict.items():
        ic_dict[key] = -math.log(value / freq_sum, 2)

    return ic_dict


def process_freq_for_ic_meddra(data_dict, meddra_dict):

    ic_dict = {}
    meddra_freq_dict = {}

    for key, value in data_dict.items():
        if key in meddra_dict:
            helper_dict = {}
            for hierarchy in meddra_dict[key]:
                for key_h, value_h in hierarchy.items():
                    if (key_h != 'RANK'):
                        helper_dict[value_h] = 0

            for term in helper_dict.keys():
                if term not in meddra_freq_dict:
                    meddra_freq_dict[term] = value
                else:
                    meddra_freq_dict[term] = meddra_freq_dict[term] + value
                    
            meddra_freq_dict[key] = value

    freq_sum = sum(list(meddra_freq_dict.values()))
    for key, value in meddra_freq_dict.items():
        ic_dict[key] = -math.log(value / freq_sum, 2)

    return ic_dict


def process_freq_for_ic_smq(data_dict, smq_dict):

    ic_dict = {}
    smq_freq_dict = {}

    for key, value in data_dict.items():
        if key in smq_dict:
            helper_dict = {}
            for hierarchy in smq_dict[key]:
                for key_h, value_h in hierarchy.items():
                    if value_h != '-':
                        helper_dict[value_h] = 0

            for term in helper_dict.keys():
                if term not in smq_freq_dict:
                    smq_freq_dict[term] = value
                else:
                    smq_freq_dict[term] = smq_freq_dict[term] + value
                    
            smq_freq_dict[key] = value

    freq_sum = sum(list(smq_freq_dict.values()))
    for key, value in smq_freq_dict.items():
        ic_dict[key] = -math.log(value / freq_sum, 2)

    return ic_dict


def preprocess_distmat(distmat):
    
    distmat_indices = {k: v for v, k in enumerate(list(distmat.keys()))}
    pre =  list(distmat.values())

    np_matrix = (np.stack(pre, axis = 0))
    np.fill_diagonal(np_matrix, -np.inf)
    
    minv = np_matrix[np_matrix > -np.inf].min()
    maxv = np_matrix[np_matrix > -np.inf].max()
    meanv = np_matrix[np_matrix > -np.inf].mean()
    normed_meanv = (meanv - minv) / (maxv - minv)
    
    distmat_props = {                    
                    'min' : minv,
                    'max' : maxv,
                    'mean' : meanv,
                    'normed_mean' : normed_meanv}
        
    return np_matrix, distmat_props, distmat_indices


def get_group_distmats_for_valid(data_dict_all, index_dict, data_dict_val, 
                                  metrics, sample_from_group):

    output = {}
    
    if (sample_from_group):
        val_react_dict = {}
        for drug_name, reactionlist in data_dict_val.items():
            for reaction in reactionlist:
                if reaction in index_dict:
                    val_react_dict[reaction] = 1
        reaction_list_all = list(val_react_dict.keys())
    else:
        reaction_list_all = list(index_dict.keys())
    
    for val_name in data_dict_val.keys():
        
        reaction_list = data_dict_val[val_name]
    
        print(str(datetime.now()) + ' - Preprocessing ' + val_name + ' for validated data query')
    
        if len(reaction_list) > 1:
            valid_reactions = {}
            random_reactions = {}
    
            for reaction in reaction_list:
                if reaction in index_dict:
                    valid_reactions[reaction] = index_dict[reaction]
                    
            if len(valid_reactions) > 1:

                for valid_reaction in valid_reactions:
                    random_reaction = random.choice(reaction_list_all)
    
                    while (random_reaction in valid_reactions or
                           random_reaction in random_reactions):
                       random_reaction = random.choice(reaction_list_all)
    
                    random_reactions[random_reaction] = index_dict[random_reaction]
                
                val_indices = list(valid_reactions.values())
                rand_indices = list(random_reactions.values())
                complete_list = val_indices + rand_indices
                output[val_name] = {}
                output[val_name]['group_indices'] = [list(range(len(val_indices))), list(range(len(val_indices), len(complete_list)))]
        
                for metric in metrics:
                    np_matrix = data_dict_all[metric]       
                    np_matrix_combined = np.take(np_matrix, complete_list, 0)
                    np_matrix_combined = np.take(np_matrix_combined, complete_list, 1)
                    output[val_name][metric] = np_matrix_combined
                    
        else: continue
                    
    return output


def transform_dict_to_np(data_dict):
    
    x = np.zeros((len(data_dict), len(data_dict[list(data_dict.keys())[0]])))
    z = []
    i = 0

    for key, value in data_dict.items():
            
        x[i, :] = value
        z.append(key)
        i += 1
                    
    return np.asarray(z), x


def separate_hier_by_tiers(hier_dict):

    tier_dicts = {}
    
    for tier in hier_dict[next(iter(hier_dict))][0].keys():
        tier_dict = {}
        tier_dict_dedup = {}
        
        for key, value in hier_dict.items():
            tier_dict_dedup[key] = {}
            
            for entry in value:
                tier_dict_dedup[key][entry[tier]] = 1
                
        for key, value in tier_dict_dedup.items():
            tier_dict[key] = ""
            sdict = sort_dict(value)
            
            for entry in sdict.keys():
                tier_dict[key] += ' | ' + entry
                
        tier_dicts[tier] = tier_dict
    
    return tier_dicts


def separate_valid_by_reacts(valid_dict):
    
    val_dict_to_reacts = {}

    for key, value in valid_dict.items():
        slist = sorted(value, key = str.lower)
        
        for entry in slist:
            if entry not in val_dict_to_reacts:
                val_dict_to_reacts[entry] = ' | ' + key
            else:
                val_dict_to_reacts[entry] += ' | ' + key
                
    return val_dict_to_reacts


def merge_atc_onto_reactions_by_valid(atc_hier_dict, atc_dicts, valid_dict):
    
    merged_dict = {}

    for tier, data in atc_dicts.items():
        merged_dict[tier] = {}
        
    for key, value in valid_dict.items():
        if key not in atc_hier_dict:
            continue
        for entry in value:
            for atc_hier in atc_hier_dict[key]:
                for atc_entry_key, atc_entry in atc_hier.items():
                    if entry not in merged_dict[atc_entry_key]:
                        merged_dict[atc_entry_key][entry] = {}
                    merged_dict[atc_entry_key][entry][atc_entry] = 1
                    
    for tier, data in merged_dict.items():
        for entry, atc_dict in data.items():
            line = ""
            sdict = sort_dict(atc_dict)
            for atc_entry in sdict.keys():
                line += ' | ' + atc_entry
            
            merged_dict[tier][entry] = line
    
    return merged_dict


#%% Define functions for corpus processing

def prep_corpus_ntx(data, n_pair_num, freqsamp, epoch):
    
    print(str(datetime.now()) + "- Building corpus for epoch number " + str(epoch))
    data_shuffled = list(data.items())
    random.shuffle(data_shuffled)
    in_data = dict(data_shuffled)
    
    in_data_listified = []
    dup_dict = {}
    
    for i, (key, value) in enumerate(in_data.items()):
        if (i % n_pair_num == 0):
            dup_dict.clear()
        dup_dict[key] = 1
        if not in_data[key].keys() <= dup_dict.keys():
            keys = list(in_data[key].keys())
            if freqsamp:
                freq_sum = sum([float(k)**(3/4) for k in in_data[key].values()])
                freq_normalized = [(float(k)**(3/4))/freq_sum for k in in_data[key].values()]
                positive = np.random.default_rng().choice(keys, p = freq_normalized)
                while (positive in dup_dict):
                    positive = np.random.default_rng().choice(keys, p = freq_normalized)
            else:
                positive = random.choice(keys)
                while (positive in dup_dict):
                    positive = random.choice(keys)
                
            dup_dict[positive] = 1
            in_data_listified.append([key, positive])
            
    return np.array(in_data_listified)


def child_proc_nsg(data, context_indices, context_unigram, numof_negs, data_size, worker_num):
    
    print(str(datetime.now()) + ' - Starting worker no. %d with %d pairs out of %d' % (
                                worker_num, len(data), data_size))
    
    in_data = data.apply(lambda row: prep_corpus_nsg(
                                   row['TINDEX'], row['CINDEX'],
                                   row['SUBS'], context_indices,
                                   context_unigram, numof_negs), axis = 1)
    
    print(str(datetime.now()) + ' - Worker no. %d has finished processing' % (worker_num))
    
    return filter(None, in_data)


def prep_corpus_nsg(target_index, context_index, subsamp_chance, 
                context_indices, context_unigram, numof_negs):
    
    if (np.random.default_rng().random() < subsamp_chance):
        in_data = []        
        in_data.append(np.array([target_index, context_index, 1]))
        for i in range(numof_negs):
            negative_sample = np.random.default_rng().choice(context_indices, p = context_unigram)
            while(negative_sample == context_index):
                negative_sample = np.random.default_rng().choice(context_indices, p = context_unigram)
                
            in_data.append(np.array([target_index, negative_sample, 0]))
    
        return in_data


#%% Define functions for classifier preprocessing

def swap_embedding_to_index(react_dict):
    
    reaction_index_dict = {}
    
    for i, (key, value) in enumerate(react_dict.items()):
        reaction_index_dict[key] = [i]
        
    return reaction_index_dict


def negative_sample_train_test_split_by_type(main_df, tt_split, use_freq_sampling, by_type, repeat):
    
    if by_type == 'DRUG':
        other_type = 'REACTION'
    elif by_type == 'REACTION':
        other_type = 'DRUG'
    else:
        sys.exit("[Error] - valid negative sampling by data types are: drug, reaction.")
        
    positive_data = main_df[main_df['LABEL'] == 1]
    
    if tt_split != 0:
        train, test = train_test_split(positive_data, test_size = tt_split)
    else:
        train = positive_data
        test = pd.DataFrame(columns=['DRUG', 'REACTION'])
    
    main_dict_train = {}
    main_dict_test = {}
    
    for index, row in train.iterrows():
        if row[by_type] not in main_dict_train:
            main_dict_train[row[by_type]] = {row[other_type]}
        else:
            main_dict_train[row[by_type]].add(row[other_type])
            
    for index, row in test.iterrows():
        if row[by_type] not in main_dict_test:
            main_dict_test[row[by_type]] = {row[other_type]}
        else:
            main_dict_test[row[by_type]].add(row[other_type])
            
    random_sampled_pairs_train = []
    random_sampled_pairs_test = []
    neg_set = set()

    if use_freq_sampling:
        subsamp_dict_drug_complete, subsamp_dict_reac_complete = generate_sampling_maps(positive_data)
        subsamp_dict_drug_train, subsamp_dict_reac_train = generate_sampling_maps(train)
        
        if by_type == 'DRUG':
            entry_list_train = list(subsamp_dict_reac_train.keys())
            entry_prob_train = list(subsamp_dict_reac_train.values())
            entry_list_test = list(subsamp_dict_reac_complete.keys())
            entry_prob_test = list(subsamp_dict_reac_complete.values())
        else:
            entry_list_train = list(subsamp_dict_drug_train.keys())
            entry_prob_train = list(subsamp_dict_drug_train.values())
            entry_list_test = list(subsamp_dict_drug_complete.keys())
            entry_prob_test = list(subsamp_dict_drug_complete.values())
        
        for i in range(repeat):
            
            for key, value in main_dict_train.items():
                
                for entry in value:
    
                    new_neg = np.random.default_rng().choice(entry_list_train, p = entry_prob_train)
                    
                    while (new_neg in value or (key + '_' + new_neg) in neg_set or (new_neg + '_' + key) in neg_set):
                        new_neg = np.random.default_rng().choice(entry_list_train, p = entry_prob_train)
                        
                    if by_type == 'DRUG':
                        random_sampled_pairs_train.append([key, entry, 1])
                        random_sampled_pairs_train.append([key, new_neg, 0])
                        
                        # These commented lines would change the sampling to keep only the uniques
                        # neg_set.add(key + '_' + new_neg)
                    else:
                        random_sampled_pairs_train.append([entry, key, 1])
                        random_sampled_pairs_train.append([new_neg, key, 0])
                        # neg_set.add(new_neg + '_' + key)
                                    
        for key, value in main_dict_test.items():
            
            for entry in value:
                
                new_neg = np.random.default_rng().choice(entry_list_test, p = entry_prob_test)
                
                while ((new_neg in value) or (key in main_dict_train and new_neg in main_dict_train[key])
                       or (key + '_' + new_neg) in neg_set or (new_neg + '_' + key) in neg_set):
                    new_neg = np.random.default_rng().choice(entry_list_test, p = entry_prob_test)

                if by_type == 'DRUG':
                    random_sampled_pairs_test.append([key, entry, 1])
                    random_sampled_pairs_test.append([key, new_neg, 0])
                    # neg_set.add(key + '_' + new_neg)
                else:
                    random_sampled_pairs_test.append([entry, key, 1])
                    random_sampled_pairs_test.append([new_neg, key, 0])
                    # neg_set.add(new_neg + '_' + key)
        
    else:
        pos_entries_train = train[other_type].unique()
        pos_entries_test = test[other_type].unique()
        
        for i in range(repeat):
            
            for key, value in main_dict_train.items():
                
                for entry in value:
                    
                    new_neg = np.random.default_rng().choice(pos_entries_train)
                    
                    while (new_neg in value or (key + '_' + new_neg) in neg_set or (new_neg + '_' + key) in neg_set):
                        new_neg = np.random.default_rng().choice(pos_entries_train)
                    
                    if by_type == 'DRUG':
                        random_sampled_pairs_train.append([key, entry, 1])
                        random_sampled_pairs_train.append([key, new_neg, 0])
                        # neg_set.add(key + '_' + new_neg)
                    else:
                        random_sampled_pairs_train.append([entry, key, 1])
                        random_sampled_pairs_train.append([new_neg, key, 0])
                        # neg_set.add(new_neg + '_' + key)
                
        for key, value in main_dict_test.items():
            
            for entry in value:
                
                new_neg = np.random.default_rng().choice(pos_entries_test)
                
                while ((new_neg in value) or (key in main_dict_train and new_neg in main_dict_train[key])
                       or (key + '_' + new_neg) in neg_set or (new_neg + '_' + key) in neg_set):
                    new_neg = np.random.default_rng().choice(pos_entries_test)
                
                if by_type == 'DRUG':
                    random_sampled_pairs_test.append([key, entry, 1])
                    random_sampled_pairs_test.append([key, new_neg, 0])
                    # neg_set.add(key + '_' + new_neg)
                else:
                    random_sampled_pairs_test.append([entry, key, 1])
                    random_sampled_pairs_test.append([new_neg, key, 0])
                    # neg_set.add(new_neg + '_' + key)
        
    false_neg_count = set()
    
    for entry in random_sampled_pairs_train:
        if (entry[2] == 0 and [entry[0], entry[1], 1] in random_sampled_pairs_test):
            false_neg_count.add(entry[0] + '_' + entry[1])
                    
    if tt_split != 0:
        false_neg_rate = len(false_neg_count)/len(random_sampled_pairs_test)
    else:
        false_neg_rate = 0
            
    return random_sampled_pairs_train, random_sampled_pairs_test, false_neg_rate
    

def negative_sample_train_test_split_full_random(main_df, tt_split, use_freq_sampling, repeat):
    
    positive_data = main_df[main_df['LABEL'] == 1]
    
    if tt_split != 0:
        train, test = train_test_split(positive_data, test_size = tt_split)
    else:
        train = positive_data
        test = pd.DataFrame()
        test = pd.DataFrame(columns=['DRUG', 'REACTION'])
            
    train_list = []
    test_list = []
            
    for index, row in train.iterrows():
        train_list.append([row['DRUG'], row['REACTION']])
        
    for index, row in test.iterrows():
        test_list.append([row['DRUG'], row['REACTION']])
    
    random_sampled_pairs_train = []
    random_sampled_pairs_test = []
        
    if use_freq_sampling:
        subsamp_dict_drug_complete, subsamp_dict_reac_complete = generate_sampling_maps(positive_data)
        subsamp_dict_drug_train, subsamp_dict_reac_train = generate_sampling_maps(train)
        
        reac_list_train = list(subsamp_dict_reac_train.keys())
        reac_prob_train = list(subsamp_dict_reac_train.values())
        drug_list_train = list(subsamp_dict_drug_train.keys())
        drug_prob_train = list(subsamp_dict_drug_train.values())
        
        reac_list_test = list(subsamp_dict_reac_complete.keys())
        reac_prob_test = list(subsamp_dict_reac_complete.values())
        drug_list_test = list(subsamp_dict_drug_complete.keys())
        drug_prob_test = list(subsamp_dict_drug_complete.values())
        
        while len(random_sampled_pairs_train) != len(train_list) * repeat:
            
            new_reac = np.random.default_rng().choice(reac_list_train, p = reac_prob_train)
            new_drug = np.random.default_rng().choice(drug_list_train, p = drug_prob_train)
            
            while (([new_drug, new_reac] in train_list) 
                   # or ([new_drug, new_reac] in random_sampled_pairs_train)
                   ):
                new_reac = np.random.default_rng().choice(reac_list_train, p = reac_prob_train)
                new_drug = np.random.default_rng().choice(drug_list_train, p = drug_prob_train)
                
            new_entry = [new_drug, new_reac, 0]
            random_sampled_pairs_train.append(new_entry)
            
        while len(random_sampled_pairs_test) != len(test_list):
            
            new_reac = np.random.default_rng().choice(reac_list_test, p = reac_prob_test)
            new_drug = np.random.default_rng().choice(drug_list_test, p = drug_prob_test)
            
            while (([new_drug, new_reac] in test_list) 
                   # or ([new_drug, new_reac] in random_sampled_pairs_test)
                   or ([new_drug, new_reac] in train_list)):
                new_reac = np.random.default_rng().choice(reac_list_test, p = reac_prob_test)
                new_drug = np.random.default_rng().choice(drug_list_test, p = drug_prob_test)
                
            new_entry = [new_drug, new_reac, 0]
            random_sampled_pairs_test.append(new_entry)
        
    else:
        pos_reacts_train = train['REACTION'].unique()
        pos_reacts_test = test['REACTION'].unique()
        pos_drugs_train = train['DRUG'].unique()
        pos_drugs_test = test['DRUG'].unique()
    
        while len(random_sampled_pairs_train) != len(train_list) * repeat:
            
            new_reac = np.random.default_rng().choice(pos_reacts_train)
            new_drug = np.random.default_rng().choice(pos_drugs_train)
            
            while (([new_drug, new_reac] in train_list) 
                   # or ([new_drug, new_reac] in random_sampled_pairs_train)
                   ):
                new_reac = np.random.default_rng().choice(pos_reacts_train)
                new_drug = np.random.default_rng().choice(pos_drugs_train)
                
            new_entry = [new_drug, new_reac, 0]
            random_sampled_pairs_train.append(new_entry)
            
        while len(random_sampled_pairs_test) != len(test_list):
            
            new_reac = np.random.default_rng().choice(pos_reacts_test)
            new_drug = np.random.default_rng().choice(pos_drugs_test)
            
            while (([new_drug, new_reac] in test_list) 
                   # or ([new_drug, new_reac] in random_sampled_pairs_test)
                    or ([new_drug, new_reac] in train_list)):
                new_reac = np.random.default_rng().choice(pos_reacts_test)
                new_drug = np.random.default_rng().choice(pos_drugs_test)
                
            new_entry = [new_drug, new_reac, 0]
            random_sampled_pairs_test.append(new_entry)
            
    for i in range(repeat - 1):
            
        for index, row in train.iterrows():
            train_list.append([row['DRUG'], row['REACTION']])
            
        for index, row in test.iterrows():
            test_list.append([row['DRUG'], row['REACTION']])
        
    for entry in train_list:
        random_sampled_pairs_train.append([entry[0], entry[1], 1])
        
    for entry in test_list:
        random_sampled_pairs_test.append([entry[0], entry[1], 1])
    
    false_neg_count = set()
    
    for entry in random_sampled_pairs_train:
        if (entry[2] == 0 and [entry[0], entry[1], 1] in random_sampled_pairs_test):
            false_neg_count.add(entry[0] + '_' + entry[1])
                    
    if tt_split != 0:
        false_neg_rate = len(false_neg_count)/len(random_sampled_pairs_test)
    else:
        false_neg_rate = 0
            
    return random_sampled_pairs_train, random_sampled_pairs_test, false_neg_rate
            
            
#%% Define functions for standardization

def standardize_simple_np(x):
    
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    
    return scaler, x_scaled

def readjust_matrix_to_distance(metric, distmat):
    
    if metric == 'cosine':
        return (distmat * -1) + 1
    # elif (metric != 'meanminpath' and metric != 'resnikmeddra'
    #       and metric != 'jiangconrathmeddra' and metric != 'linmeddra'
    #       and metric != 'resniksmq' and metric != 'jiangconrathsmq'
    #       and metric != 'linsmq' and metric != 'primaryminpath'
    #       and metric != 'meanpath' and metric != 'primarypath'
    #       and metric != 'smqmeanminpath'):
    else:
        return (distmat * -1) + distmat.max()




    