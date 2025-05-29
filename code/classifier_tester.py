#%% Import libraries

import sys
import os
import time

import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa

from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from joblib import load

# Project specific modules
from lib import io_file_handling as io
from lib import preprocessing as pp


#%% Tune CPU for better tensorflow parallelization
# Even if the model is run on GPU, these parameters reduce the CPU bottleneck during training
# https://www.intel.com/content/www/us/en/developer/articles/technical/effectively-train-execute-ml-dl-projects-on-cpus.html
# Additionally, switch to Tensorflows's graph mode, instead of eager, as the latter heavily leaks memory during Keras' model.fit()

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.set_soft_device_placement(enabled = True)
tf.compat.v1.disable_eager_execution()

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity = fine, verbose, compact, 1, 0"


#%% Define main class

class ClassifierTester():
    
    def __init__(self, input_settings):
        self.input_settings = input_settings
        self.read_settings()
        
        
    def read_settings(self):
        print(str(datetime.now()) + ' - Reading input parameters')
        
        # Source files
        self.reaction_embedding_path = self.input_settings["oSourceList"]["oReactionEmbedding"]
        self.drug_features_chem_path = self.input_settings["oSourceList"]["oDrugFeaturesChem"]
        self.drug_features_bio_path = self.input_settings["oSourceList"]["oDrugFeaturesBio"]
        self.test_data_path = self.input_settings["oSourceList"]["oReferenceDatasets"]
        self.pretrained_model_path = self.input_settings["oSourceList"]["oPretrainedModel"]
        self.standard_scaler_path = self.input_settings["oSourceList"]["oStandardScaler"]
        
        # General parameters
        self.data_is_labeled = self.input_settings["oParameterList"]["oDataIsLabeled"]

        # Data processing parameters
        self.no_emb = self.input_settings["oParameterList"]["oNoEmbedding"]
        self.standardize_reacts = self.input_settings["oParameterList"]["oStandardizeReactions"]
        self.standardize_drugs = self.input_settings["oParameterList"]["oStandardizeDrugs"]

        # Output path
        self.output_path = self.input_settings["oOutputPath"]
        self.project_name = self.input_settings["oProjectName"].lower()
        
        self.out_dir = io.make_out_dir(self.output_path, self.project_name)
        self.out_csv = self.input_settings["oOutCsv"]


    def test(self):
        
        self.read_files()
        self.process_files()
        self.test_model()
        
        
    def read_files(self):
        print(str(datetime.now()) + ' - Reading input files')
        
        self.drug_dict_chem = io.read_vector_dict(self.drug_features_chem_path, '|', False)
        self.drug_dict_bio = io.read_vector_dict(self.drug_features_bio_path, '|', False)
        self.reaction_dict = io.read_vector_dict(self.reaction_embedding_path, '|', False)
        self.test_dict = {}
        
        for test_file in self.test_data_path:
            io.read_test_set(test_file, '|', False, self.test_dict, labelled = self.data_is_labeled)
            
        self.scaler = load(self.standard_scaler_path)
        
        
    def process_files(self):
        print(str(datetime.now()) + ' - Processing input files')
        
        # Filter out empty drug-feature entries (caused by faulty data preparation)
        for key in list(self.drug_dict_chem.keys()):
            if len(self.drug_dict_chem[key]) == 0:
                del self.drug_dict_chem[key]
                del self.drug_dict_bio[key]
                
        if self.no_emb:
            self.reaction_dict = pp.swap_embedding_to_index(self.reaction_dict)
            
        self.test_data, self.test_entries = self.prepare_data()
        
        
    def prepare_data(self):
        
        sampled_test = self.prep_test_set()                        

        test_emb_np = np.stack([self.reaction_dict[x[1]] for x in sampled_test])
        test_chem_np = np.stack([self.drug_dict_chem[x[0]] for x in sampled_test])
        test_bio_np = np.stack([self.drug_dict_bio[x[0]] for x in sampled_test])
        
        self.chem_len = test_chem_np.shape[1]
        self.bio_len = test_bio_np.shape[1]
        self.emb_len = test_emb_np.shape[1]
        
        print(str(datetime.now()) + ' - Standardizing features')
        # Calculate standardization scaler for the train data, apply it to both train and test data
        
        if self.standardize_reacts and self.standardize_drugs and not self.no_emb:
            scaled_test = self.scaler.transform(np.concatenate((test_emb_np, test_chem_np), axis = 1))
            test_data = np.concatenate((scaled_test, test_bio_np), axis = 1)
            
        elif self.standardize_reacts and not self.no_emb:
            scaled_test = self.scaler.transform(test_emb_np)
            test_data = np.concatenate((scaled_test, test_chem_np, test_bio_np), axis = 1)
            
        elif self.standardize_drugs:
            scaled_test = self.scaler.transform(test_chem_np)
            test_data = np.concatenate((test_emb_np, scaled_test, test_bio_np), axis = 1)
            
        else:
            test_data = np.concatenate((test_emb_np, test_chem_np, test_bio_np), axis = 1)

        # Match final datasets with their corresponding truth labels                
        test_data = np.append(test_data, np.asarray([x[2] for x in sampled_test]).reshape((-1, 1)), axis = 1)

        return test_data, sampled_test
     
        
    def test_model(self):
        # Run a prediction on the test set, write the results into a CSV and save the model
        
        self.model = load_model(self.pretrained_model_path)
        test_preds = self.model.predict([self.test_data[:, 0:self.chem_len], self.test_data[:, self.chem_len:self.chem_len + self.bio_len], 
                                         self.test_data[:, self.chem_len + self.bio_len:-1]])
        if self.data_is_labeled:          
            final_pred = list(test_preds[:, 0])
            final_truth = self.test_data[:, -1]
            
            auroc = roc_auc_score(final_truth, final_pred)
            precision, recall, thresholds = precision_recall_curve(final_truth, final_pred)
            auprc = auc(recall, precision)
            acc = accuracy_score(final_truth, [round(x) for x in final_pred])
            
            print(str(datetime.now()) + ' - test_accuracy=%.4f, test_auroc=%.4f, test_auprc=%.4f' %(acc, auroc, auprc))
            
            if self.out_csv != "":
                next_row = [acc, auroc, auprc, self.test_data_path, self.pretrained_model_path]
                io.write_by_append(self.out_csv, next_row)
        
        test_preds_with_names = []
        
        for i in range(len(test_preds)):
            entry = self.test_entries[i]
            entry.extend(test_preds[i])
            test_preds_with_names.append(entry)
            
        io.write_list_of_lists(self.out_dir, 'test_data_with_predictions', test_preds_with_names, '|')
    
    
    def prep_test_set(self):
        
        test_list = []
        
        if self.data_is_labeled:
            
            for value in self.test_dict.values():
                if value[0] in self.drug_dict_chem and value[1] in self.reaction_dict:
                    entry = [value[0], value[1], value[2]]
                    test_list.append(entry)
        else:
            
            for value in self.test_dict.values():
                if value[0] in self.drug_dict_chem and value[1] in self.reaction_dict:
                    entry = [value[0], value[1]]
                    test_list.append(entry)
                
        return test_list
        
    
    def get_out_dir(self):
        return self.out_dir


#%% Initiate main sequence

if __name__ == "__main__":
    
    # Start timer for computation time measurements
    start_time = time.time()
    # Read the provided json file from command-line arguments
    input_settings = io.read_json_input(sys.argv[1])
    # Initialize the main class
    tester = ClassifierTester(input_settings)
    # Initialization of the class creates the output directory, where the input json file can now be copied
    io.copy_to_out_dir(tester.get_out_dir(), sys.argv[1], 'input.json')
    # Redirect print() and errors so that it will also write into a log.txt at out_dir
    sys.stdout = io.Logger(sys.stdout, tester.get_out_dir())
    sys.stderr = sys.stdout
    # Begin model training, which includes data reading and preprocessing
    tester.test()
    # Conclude
    print(str(datetime.now()) + ' - All done!')
    elapsed_time = time.time() - start_time
    print(str(datetime.now()) + " - Only took %02d seconds!" % (elapsed_time))

