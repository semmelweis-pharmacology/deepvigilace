#%% Import libraries

import sys
import os
import csv
import json
import shutil

import numpy as np
import pandas as pd
import itertools as it

from datetime import datetime
from pathlib import Path


#%% Define class for general logging

class Logger():

    def __init__(self, stream, out_dir):
        self.stream = stream
        self.filename = open(out_dir + 'log.txt', 'w')

    def write(self, message):
        self.stream.write(message)
        self.filename.write(message)            
        self.filename.flush()
       
    def flush(self):
        self.stream.flush()
    

#%% Define functions for input file handling

def read_json_input(path):
    
    try:
        with open(path) as jsonfile:
            infile = json.load(jsonfile)
            
        return infile
    
    except Exception as e:
        print('[Error] Input json could not be opened.')
        sys.exit(e)


def read_simple_dict(path, delimiter, header, index1 = 0, index2 = 1):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            for line in reader:
                dictionary[line[index1]] = float(line[index2])
                
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)


def read_simple_naming_dict(path, delimiter, header, index1 = 0, index2 = 1):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            for line in reader:
                dictionary[line[index1].upper()] = str(line[index2])
                
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)
        

def read_vector_dict(path, delimiter, header):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            for line in reader:
                dictionary[line[0]] = np.asarray(line[1:], dtype = np.float64)
                
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)


def read_valid_pair_dict(path, delimiter, header):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            for line in reader:            
                if line[0] not in dictionary:
                    dictionary[line[0]] = []
                    
                dictionary[line[0]].append(line[1])
                
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)
        
        
def read_test_set(path, delimiter, header, dictionary, labelled = True):
    
    try:
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            if labelled:
                for line in reader:
                    combname = line[0] + '_' + line[1]
                    dictionary[combname] = [line[0], line[1], int(line[2])]
            else:
                for line in reader:
                    combname = line[0] + '_' + line[1]
                    dictionary[combname] = [line[0], line[1]]
                
    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)
        
        
def read_valid_pair_dicts_merged(path, delimiter, header):
    
    try:
        validated_dict = {}
        
        for val_dict in path:
            val_pairs = read_valid_pair_dict(val_dict, delimiter, header)
            
            for key, value in val_pairs.items():
                if key not in validated_dict:
                    validated_dict[key] = value
                    
                else:
                    
                    for reac in value:
                        if reac not in validated_dict[key]:
                            validated_dict[key].append(reac)
                            
        return validated_dict
                            
    except Exception as e:
        print("[Error] " + path + "files could not be opened.")
        sys.exit(e)


def read_meddra_hierarchy_meshup(path, delimiter, header, prime_only):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)       
            if header: next(reader, None)
            for line in reader:
                if (prime_only and line[10] != 'PRIMARY'):
                    continue
                
                if (line[2] in dictionary):
                    for i in range(2, 10, 2):
                        dictionary[line[2]][line[i]] = line[i + 1]
                        dictionary[line[2]][line[i + 1]] = line[i]

                else:
                    dict_entry = {}
                    for i in range(2, 10, 2):
                        dict_entry[line[i]] = line[i + 1]
                        dict_entry[line[i + 1]] = line[i]

                    dictionary[line[2]] = dict_entry
                    
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)


def read_meddra_hierarchy_separate(path, delimiter, header, prime_only):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)       
            if header: next(reader, None)
            for line in reader:
                if (prime_only and line[10] != 'PRIMARY'):
                    continue
                dict_entry = {}
                dict_entry['HLT'] = line[4]
                dict_entry['HLGT'] = line[6]
                dict_entry['SOC'] = line[8]
                dict_entry['RANK'] = line[10]
                if (line[2] not in dictionary):
                    dictionary[line[2]] = []
                if (dict_entry not in dictionary[line[2]]):
                    dictionary[line[2]].append(dict_entry)
                    
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)


def read_5level_hierarchy_meshup(path, delimiter, header):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            for line in reader:                
                if (line[0] in dictionary):
                    for i in range(2, 12, 2):
                        dictionary[line[0]][line[i]] = line[i + 1]
                        dictionary[line[0]][line[i + 1]] = line[i]

                else:
                    dict_entry = {}
                    for i in range(2, 12, 2):
                        dict_entry[line[i]] = line[i + 1]
                        dict_entry[line[i + 1]] = line[i]

                    dictionary[line[0]] = dict_entry
                    
        return dictionary

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)


def read_5level_hierarchy_separate(path, delimiter, header, name):
    
    try:
        dictionary = {}
        with open(path, "r") as infile:
            reader = csv.reader(infile, delimiter = delimiter)
            if header: next(reader, None)
            for line in reader: 
                dict_entry = {}
                for i in range(1, 6):
                    dict_entry[name + str(6 - i)] = line[i * 2]

                if (line[0] not in dictionary):
                    dictionary[line[0]] = []
                    
                dictionary[line[0]].append(dict_entry)
                    
        return dictionary                

    except Exception as e:
        print("[Error] " + path + " could not be opened.")
        sys.exit(e)


#%% Define functions for output file handling

def write_simple_dict(path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            for key, value in dictionary.items():
                line = []
                line.append(key)
                line.append(value)
                writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
            

def write_vector_dict(path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            for key in dictionary.keys():
                line = []
                line.append(key)
                for value in dictionary[key]:
                    line.append(value)
                writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        
        
def write_list_of_lists(path, name, list_of_lists, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            for entry in list_of_lists:
                line = []
                for value in entry:
                    line.append(value)
                writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        
        
def write_by_append(full_path, data_list):
    
    try:
        with open(full_path, 'a') as outfile:  
            writer = csv.writer(outfile, delimiter='|')
            line = []
            for data in data_list:
                line.append(data)
            writer.writerow(line)
            
    except Exception as e:
        print("[Error] " + full_path + " could not be written.")
        sys.exit(e)
            

def write_similarity_matrix(path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            first_line = []
            first_line.append('')
            for key in dictionary.keys():
                first_line.append(key)
                
            writer.writerow(first_line)
            for key in dictionary.keys():
                line = []
                line.append(key)
                for value in dictionary[key].values():
                    line.append(value)
                    
                writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        

def write_top_ranking(path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            first_line = []
            for key in dictionary.keys():
                first_line.append(key)
                first_line.append('')
                
            writer.writerow(first_line)
            for i in range(len(dictionary[next(iter(dictionary))])):
                line = []
                for key in dictionary.keys():
                    line.append(dictionary[key][i][0])
                    line.append(dictionary[key][i][1])
                    
                writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        

def write_validity_comparison_all(metrics, path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            first_line = []
            first_line.append('')
            for metric in metrics:
                first_line.append(metric)
                
            writer.writerow(first_line)
            for key, value in dictionary.items():
                for key_in, value_in in value.items():
                    line = []
                    line.append(key + '_' + key_in)
                    for val in value_in:
                        line.append(val)
                        
                    writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)     
           

def write_validity_comparison_single(metrics, path, name, dictionary, column, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            first_line = []
            first_line.append('')        
            for metric in metrics:
                first_line.append(metric)
                
            writer.writerow(first_line)
            for key, value in dictionary.items():
                line = []
                line.append(key)
                for val in value[column]:
                    line.append(val)
                    
                writer.writerow(line)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        

def write_validity_postprocess(path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
            first_line = []
            first_line.append('')
            for key in dictionary[next(iter(dictionary))].keys():
                first_line.append(key)
                
            writer.writerow(first_line)
            for key, value in dictionary.items():
                line = []
                line.append(key)
                for key_in, value_in in value.items():
                    line.append(value_in)
                    
                writer.writerow(line)
                
        transpose_csv_with_pandas(path + name + ".csv", delimiter)
                
    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        

def write_validity_percentile(path, name, dictionary, delimiter):
    
    try:
        with open(path + name + ".csv", 'w+') as outfile:  
            writer = csv.writer(outfile, delimiter = delimiter)
    
            for key, value in dictionary.items():
                line_1 = []
                line_2 = []
                line_1.append(key)
                line_2.append('')
                for key_in, value_in in value.items():
                    line_1.append(str(key_in))
                    line_2.append(value_in)
                writer.writerow(line_1)
                writer.writerow(line_2)
                
        transpose_csv_without_pandas(path + name, delimiter)

    except Exception as e:
        print("[Error] " + name + ".csv could not be written.")
        sys.exit(e)
        

def print_embeddings(target_layer, context_layer, epoch_num,
         target_dict, context_dict, out_dir):
    
    print(str(datetime.now()) + " - Printing embedding of epoch {}".format(epoch_num))
    
    target_dict_new = {}
    weights = target_layer.get_weights()[0]
    for i, key in enumerate(target_dict.keys()):
        target_dict_new[key] = weights[i]
        
    context_dict_new = {}
    weights = context_layer.get_weights()[0]
    for i, key in enumerate(context_dict.keys()):
        context_dict_new[key] = weights[i]
        
    write_vector_dict(out_dir, str(epoch_num) + "_target_dict", target_dict_new, '|')
    write_vector_dict(out_dir, str(epoch_num) + "_context_dict", context_dict_new, '|')
        
    return None


def transpose_csv_with_pandas(filename, delimiter):
    
    try:
        pd.read_csv(
            filename, header = None, delimiter = delimiter, dtype = str).T.to_csv(
            filename, header = False, index = False, sep = delimiter, na_rep='')
                
    except Exception as e:
        print("[Error] " + filename + 
              " could not be transposed. Might be because of inconsistent row length in the original.")
        sys.exit(e)


def transpose_csv_without_pandas(filename_without_ext, delimiter):

    try:
        os.rename(filename_without_ext + '.csv', filename_without_ext + '_p.csv')
        with open(filename_without_ext + '_p.csv') as infile, open(filename_without_ext + '.csv', 'w') as outfile: 
            csv.writer(outfile, delimiter = delimiter).writerows(
                it.zip_longest(*csv.reader(infile, delimiter = delimiter)))
        remove_file(filename_without_ext + '_p.csv')
        
    except Exception as e:
        print("[Error] " + filename_without_ext + ".csv could not be transposed.")
        sys.exit(e)


def copy_to_out_dir(out_path, file, new_name = None):
    
    if new_name == None:
        new_name = Path(file).name
    
    try:
        shutil.copyfile(file, out_path + new_name)
    except Exception as e:
        print('[Error] ' + file + ' could not be copied.')
        print(e)
        pass

def clear_file(filename):
    
    try:
        open(filename, 'w').close()
    except Exception as e:
        print('[Error] ' + filename + ' could not be cleared.')
        print(e)
        pass
    
def remove_file(filename):
    
    try:
        os.remove(filename)
    except OSError:
        pass


#%% Define functions for directory handling

def make_out_dir(path, name):
    
    try:
        time = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        Path(path + name + '_' + time).mkdir(parents = True, exist_ok = True)
        
        return path + name + '_' + time + '/'
    
    except Exception as e:
        print('[Error] Output directory could not be created.')
        sys.exit(e)
        
        
def make_dir(path, name):
    
    try:
        Path(path + name).mkdir(parents = True, exist_ok = True)
        
        return path + name + '/'
    
    except Exception as e:
        print('[Error] Output directory could not be created.')
        sys.exit(e)
        
