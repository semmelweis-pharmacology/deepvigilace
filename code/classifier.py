#%% Import libraries

import sys
import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from joblib import dump

# Project specific modules
from lib import io_file_handling as io
from lib import preprocessing as pp
from lib import models


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

class Classifier():
        
    """
    The Classifier class encapsulates the training of a fully connected neural 
    network that performs classification of drug-ADR pairs. There are different
    training regimens available based on negative data generation, and 
    performance can be evaluated in either a train-test split manner or with 
    provided reference data, as well as it can be compared to ROR results.

    ...
    
    Args
    ---------
    input_settings: json object from the input file, using json.load()
        The input JSON file defines the location of the necessary source files, 
        the path to the output directory and the changeable hyperparameters for
        the data processing and training. See the template JSON and the class 
        attributes for further details.

    Attributes
    ----------
    input_settings: json object, see in Args
    
    main_data_path: str
        Full path to the main input data file (CSV), containing a list 
        (delimiter '|', no header) of unique drug-ADR pairs. It consists 
        of six columns (ID, DRUG, REACTION, LABEL, FREQ, ROR) where the first 
        one holds a unique ID merged from the individual names of the pairs, 
        the second includes the drug name, the third includes the name of the
        reaction, in the fourth there is the label of the pair: -1 means
        unlabeled (from adverse event reports), 0 means negative association 
        (only present in test data), 1 means positive association (from 
        validated drug-ADR data like drug labels). The fifth includes the 
        number of occurrences (int) of the given pair as seen in the source 
        data. The sixth includes the lower confidence interval value (float) of
        the ROR (reporting odds ratio) as calculated based on the source data. 
        Drug names are expected to be in DrugBank canonic names, and reactions 
        in MedDRA PTs, so that they can be assigned features from the feature 
        dictionaries. See reaction_embedding_path, drug_features_chem_path and 
        drug_features_bio_path for further details. Pairs with 0 occurrence 
        and -1 as a (placeholder) ROR lower bound value may be present, given
        the pair was not present in the reporting data, only in the validated 
        drug-ADR data. The utilization of this main data heavily depends on the
        selected training regimen. See negative_sample for further details.
        
    reaction_embedding_path: str
        Full path to the reaction embedding data file (CSV, delimiter '|', no 
        header), containing in the first column the MedDRA PT names of the 
        embedded adverse events, while other columns are populated by the 
        elements of the corresponding embedding vectors, containing one 
        element (float) per cell. Consequently, this file is the product of
        the embedding module. Matching of the training data with these 
        embedding vectors is done through their MedDRA PT names.
        
    drug_features_chem_path: str
        Full path to the drug feature data file (CSV, delimiter '|', no 
        header), containing in the first column the DrugBank canonic name of 
        the given drug substance, while other columns are populated by its 
        chemical features from the PubChem database in float and integer 
        formats. Consequently, this file is the product of the data processing
        module, where DrugBank substances without connected PubChem compound
        profiles are thus filtered out. Matching of the training data with 
        these feature vectors is done through their DrugBank canonic names.
        
        Disclaimer, there appear to be empty rows in this file, that are the 
        result of inproper filtering. Some PubChem compound IDs are no longer
        actually searchable (status: non-live), meaning that even though the 
        DrugBank profile was linked to a PubChem profile, the download query 
        did not return it, and the data processing module did not check whether
        all CID were successfully downloaded or not. Because of this, there are
        additional measures in this code to filter out drugs without features.
        
    drug_features_bio_path: str
        Full path to the drug feature data file (CSV, delimiter '|', no 
        header), containing in the first column the DrugBank canonic name of 
        the given drug substance, while other columns are populated by its 
        biological features containing 0s and 1s that correspond to whether 
        that given protein is assigned to the substance as either a 
        transporter, enzyme, carrier or target in the DrugBank database. 
        Consequently, this file is the product of the data processing module. 
        The proteins are left unnamed with no further details provided other
        than these multilabel binary vectors. Matching of the training data 
        with these features is done through their DrugBank canonic names.
        
    reference_data_path: list of str
        A list of full paths to the reference drug-reaction dataset files (CSV,
        delimiter '|', no header). These contain three columns (DRUG, REACTION,
        LABEL), where similarly to the main_data_path, drug names are assumed 
        to be in their DrugBank canonic form and reactions to be MedDRA PTs so 
        they can be matched with the corresponding feature vectors. The label 
        column contains 1s (positive association) and 0s (negative association)
        as determined for the corresponding pair in the source database. 
        Repeating pairs are merged together during reading and contradicting 
        labels are NOT checked, only the last one encountered is kept. These 
        reference files are completely ignored when tt_split is set to larger 
        than 0, as then a train-test split according to the given ratio is 
        used. Consequently, setting tt_split to 0 will require at least one 
        reference dataset to be provided for performance evaluation. When these
        are used, the positive, validated pairs (label of 1) of the main 
        training data are removed if they are also present in any of the 
        reference sets to ensure less bias in the estimation of the 
        prerformance of the classifier (its ability to predict the reference
        pairs as a test would otherwise be boosted by the fact it already has
        seen some of them).
        
    pretrained_model_path: str
        Full path of a pretrained classifier model file (H5) that has the same
        basic structure (number and length of input vectors, and the output) as
        the one trained by this module. This makes the code skip the model
        building part and instead will train the pretrained model with the
        provided parameters and data, which enables fine-tuning or resuming
        training with a different configuration.
        
    calc_ror_performance: boolean
        Determines whether to calculate the performance of the ROR lower-bound
        confidence values on the input (and if available, on the reference) 
        data. See get_ror_performance() for further details.
        
    no_emb: boolean
        Determines whether the model should use the provided embedding vectors
        or not. When set to true, the dictionary containing the reaction
        features is swapped to a simple index table, and then a special model
        is built, which expects index integers for the reactions. These indices
        are mapped into a one-hot vector representation inside the model, and
        embedded during the training of the classifier. This alternative 
        approach serves to demonstrate the performance differences the embedded
        features can create. In the fine-tuning of a pretrained model and the 
        classifier tester module, it is NOT checked whether the provided 
        model is the no_emb type or not, it has to be set to the correct one 
        manually.

    negative_sample: boolean
        Determines whether to apply one of the negative random sampling methods
        on the training data or not.
        
        In case its false:
            The provided input data is expected to contain unlabled pairs (-1)
            from adverse event reports, and validated pairs (1) from sources
            such as drug labels. With this training regimen, all unlabeled 
            pairs are considered to be in the negative class (0), while 
            validated pairs remain the positive class. The result is an 
            unbalanced training data, on which the performance of the model is
            overestimated by the AUROC metric because of the majority negative
            class. The AUPRC metric is thus more telling, which reflects the
            model's struggle of correctly classifying the minority positive 
            class.
            
        In case its true:
            The provided input data only needs to contain validated pairs (1)
            from sources such as drug labels, the others are discarded. Then,
            one of the 6 available negative sampling methods is applied to
            create a number of negative pairs. See negsamp_by_type, 
            negsamp_use_freq and repeat_sampling for further details. When
            tt_split is set to larger than 0, meaning the input data is divided
            to a training and testing set, the negative sampling always takes 
            place after split, and separately for the train and test sets. The 
            sampled random negatives of the train set are not checked against 
            the test set, only from the other way around. This might result 
            training negatives that are positives in the test set, but that
            reflects a real life training scenario and negates the possibility
            of a data leak.
            
    negsamp_by_type: string
        Possible values include: none, drug, reaction.
        
        In case of none:
            To create a negative pair, both a new drug and a new reaction is
            randomly chosen. For the training set, the new pairs are recombined
            from the drugs and reactions present in the train set, and are 
            checked against already existing positive pairs only, meaning this
            is sampling with replacement. If tt_split is set to larger than 0,
            meaning a test set is also created, then for those negatives the
            pairs are recombined from both the train and test sets, and the
            new negatives are checked against all entries (positive and 
            negative) of the train set and the positives of the test set. This
            sampling process is done 'size of the sets' number of times, so
            the resulting data will have label balance.
            
        In case of drug:
            The sampling process is done by looping through the positive pairs,
            and for each one we keep the drug and sample a new reaction for it.
            Other steps are the same. This results in balanced distribution of
            drugs in the positive and negative sets.
            
        In case of reaction:
            The sampling process is done by looping through the positive pairs,
            and for each one we keep the reaction and sample a new drug for it.
            Other steps are the same. This results in balanced distribution of
            reaction in the positive and negative sets.

    negsamp_use_freq: boolean
        Determines whether the sampling of new drugs and reactions for the 
        negative pairs should follow a uniform random distribution or the
        frequency distribution of the individual entities. When set to true,
        a frequency dictionary is calculated for both the drugs and reactions
        based on their number of occurrences in the provided validated data.
        These dictionaries are calculated for the train set based on only the
        train split of the data, while for the test set both the train and test
        split are used. When negative sampling is done by using these frequency
        dicitonaries, it results in more frequent drugs and reactions having a
        higher chance to appear as negative samples. It creates overall harder
        negatives, better resembling the distribution of real data.

    repeat_sampling: int
        Determines how many times the negative sampling step will be repeated.
        This is an oversampling method, where the positive samples will be 
        repeated exactly the number of times provided by this parameter, but
        the negatives will be sampled again, with replacement. It serves to
        create a better coverage of the possible negative samples, decreasing
        the variance in the model's performance throughout subsequent trainings
        as it will be less reliant on a 'lucky roll'. This also means the
        size of overall training data per epoch will be equal to:
        (size-of-train-split * 2) * repeat_sampling
        Both the resulting train and test datasets are written out into CSV
        files in the output directory.

    standardize_reacts: boolean
        If set to true, the embedding vectors of the reactions are standardized
        to have 0 mean and 1 variance. This step is done separately for the
        training and test set, where the scaling parameters calculated for the
        training set are applied on the test set as well. The scaler object is
        written out into the output directory so it can be reused in the
        classifier tester module for inference.
        
    standardize_drugs: boolean
        If set to true, the chemical feature vectors of the drugss are 
        standardized to have 0 mean and 1 variance. This step is done 
        separately for the training and test set, where the scaling parameters 
        calculated for the training set are applied on the test set as well. 
        The scaler object is written out into the output directory so it can be
        reused in the classifier tester module for inference.

    tt_split: float
        Determines the size ratio of the train-test split, such as 0.8 
        corresponds to 80% of the input data being randomly selected as the
        training set and the remaining 20% will be regarded as the testing set.
        When set to 0, no train-test split is performed, all the input data is
        used for training, and additional reference data is expected for the
        measuring of the performance of the model. See reference_data_path for
        further details.
        
    optimizer: str
        Possible values: rmsprop, sgd, adagrad, adam
        This parameter determines the type of optimizer to be used during model
        training. Based on empirical tests, the adam (Adaptive Moment 
        Estimation) optimizer is recommended.
        
    learning_rate: float
        The rate of learning passed down to the optimizer. Recommended: 0.001
        
    numof_epochs: int
        This determines the number of times the model should go through the
        entire available training data during learning. See out_csv for further
        details.
        
    batch_size: int
        Size of batches the data is fed in to the model during learning. Based 
        on available GPU memory, recommended: 256-512

    output_path: str
        Full path to the output directory (ending in '/'), where a new folder 
        will be created, named by the project_name + a time stamp.
        
    project_name: str
        Used for the creation of the output folder, can be empty.
    
    out_dir: str
        The output folder used to print out the result files. See project_name
        and output_path for further details.
        
    out_csv: str
        Full path to a CSV file outside of the out_dir folder. This is meant to
        be used for shared performance comparison throughout multiple different
        runs, such as for parameter grid search. Performance metrics are
        evaluated on either the test split or the provided reference data at 
        every 10th epochs, then written out. When no path is given, this is 
        only written to the console terminal.
        
    grid_params: dict
        A dictionary to catalogue parameters of interest for shared runs. The
        keys of this dictionary should be the intended header names for the
        shared CSV (out_csv), while the values should be the corresponding
        parameters themselves. These values will be written into the shared
        CSV, along with the performance metrics at every 10th epoch. Example:

        self.grid_params = {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            }

    main_df: pandas dataframe
        This variable stores the main input data. See main_data_path for 
        further details. It is later extended with additional data, see 
        process_files() for that.
        
    drug_dict_chem: dictionary
        A dictionary containing the chemical features of drugs. See 
        drug_features_chem_path for further details.
        
    drug_dict_bio: dictionary
        A dictionary containing the biological features of drugs. See 
        drug_features_bio_path for further details.
        
    reaction_dict: dictionary
        A dictionary containing the embedded features of reactions. See 
        reaction_embedding_path for further details.
        
    reference_dict: dictionary
        This contains the merged entries from the provided reference dataset.
        The combined string of the drug and reaction names serve as the key and
        and lists of [drug, reaction, label] are the values. See 
        reference_data_path for further details.
        
    reference_data: list of lists
        A transformed version of the reference_dict, having its values as a
        list of lists [drug, reaction, label] for easier handling in later
        functions, and also filtered by removing entries without corresponding
        drug or reaction features. It is also used to check whether reference 
        data was successfully provided, as it will be set to NONE otherwise. 
        See prep_reference_set() for further details.
        
    train_data: ndarray
        This is the output of the preprocessing steps, including the optional
        train-test split, negative sampling and standardization steps. It 
        contains an NxR numpy matrix, where N is the number of training samples
        and R is the number of the combined size of the embedded reaction
        vector, the chemical and the biological drug feature vector + 1 (the
        ground truth label). The data in this form is ready to be fed to the
        classifier model.
        
    test_data: ndarray
        It has the same basic structure as the train_data, only it either
        originates from a train-test split or the provided reference datasets.
        See tt_split and reference_data for further details.
        
    test_entries: list of lists
        This variable serves a supporting role, so that the predictions given
        to the test (or reference) data during the end of the training can
        be written into an output CSV file with names. It contains in list
        form [drug, reaction, label] the entries of the data used for 
        evaluation.

    ror_results_dict_train: dictionary
        A simple dictionary of evaluation metric name - result pairs calculated
        for the training dataset. These are written out into the output 
        directory as a CSV file. See get_ror_performance() for further details.
           
    ror_results_dict_test: dictionary
        A simple dictionary of evaluation metric name - result pairs calculated
        for the testing (or reference) dataset. These are written out into the 
        output directory as a CSV file. See get_ror_performance() for further 
        details.
    
    chem_len: int
        This variable stores the detected length of the chemical feature 
        vectors of drugs to be later passed down for the classifier model.
        
    bio_len: int
        This variable stores the detected length of the biological feature 
        vectors of drugs to be later passed down for the classifier model.
        
    emb_len: int
        This variable stores the detected length of the embedded feature 
        vectors of reactions to be later passed down for the classifier model.
        
    model: tf.keras.Model() object
        The constructed neural network model based on the input 
        hyperparameters. A simplified visualization of its structure is 
        generated in the out_dir by graphviz.


    Methods
    ----------
    read_settings():
        Extracts the input parameters from the provided json object into class
        variables. Furthermore, creates the output folder at the designated
        destination. Executed during class initialization. See out_dir and the 
        rest of the parameter descriptions for further details.

    train():
        The main method of the Classifier class, intended to start the entire
        process.
        
    read_files():
        Reads into memory the input data files provided, see main_data_path,
        reaction_embedding_path, drug_features_chem_path, 
        drug_features_bio_path and reference_data_path for further details.
        
    process_files():
        Processes the input files into their final form. The steps include
        filtering out entries which lack any features (due to flawed data
        processing), matching the entries with their corresponding drug 
        features and with either their corresponding embedded reaction features
        or reaction indices (when no_emb = True), adding new ROR related 
        columns (when calc_ror_performance = True), preprocessing the reference
        set (when provided) by launching prep_referebce_set(), then the 
        performance of the ROR metric is calculated for it (when enabled).
        Additionally, the positive class entries of the main_df that are 
        present in the reference set, with any label, are also filtered out 
        here. Finally, this method provides a brief summary of the data and 
        launches prepare_data() which performs further data processing 
        according to the selected training regimen. See below for further 
        details.
        
    prepare_data():
        This method is responsible for preparing the training and testing data
        according to the selected training regimen. The steps include the 
        train-test split of the data (when tt_split > 0), the creation of
        random negative samples (when negative_sample = True), the 
        initialization of the ROR metric calculation for the train and test
        (when tt_split > 0) datasets (when calc_ror_performance = True), and 
        finally the standardization of the features (see standardize_reacts and 
        standardize_drugs). Human-readable CSV files are generated in the 
        output folder, containing the resulted training and testing datasets.
        The output of this method is numpy matrices ready to be used for the
        training and testing of the model (see train_data and test_data) and a 
        list of lists containing the test entries by name (see test_entries).
        
    build_model():
        Builds and compiles the neural network model according to the input
        parameters, then calls visualize() to generate a simplified PNG image 
        of its structure. Alternatively, it reads in a pretrained model (when
        provided).
        
    visualize():
        Prints a summary onto the console and a simplified PNG image in the 
        output folder using graphviz.
        
    train_model():
        Initiates a logger CSV file that will store the training performance
        metrics and loss by each epoch. Then, it commences the training of the
        classifier model, running an evaluation after every 10th epoch, the 
        result of which are written out into a shared CSV file, when a path
        for that is given. See out_csv for further details.
        
    test_and_save():
        This method is executed only at the end of the training. It runs one
        final prediction on the test (or reference) dataset and writes out
        the results, along with the ground truth, into a human-readable CSV
        file. Additionally, it also saves the model as H5 file.        

    get_optimizer():
        Selects and returns a tf.keras.Model() compatible optimizer object,
        based on the input parameters.
        
    get_ror_performance(input_df, d_type, pair_threshold = 3, 
                        bound_threshold = 1, partial = None):
        
        This method is responsible for the calculation of the performance 
        metrics for the ROR approach, when calc_ror_performance is true. The
        input data (main_df) is expected to contain the ROR lower-bound 
        confidence values of each drug-reaction pair, which are slightly pre-
        processed by the process_files() method. By default, the evaluation 
        metric calculation prepares a confusion matrix according to EMA 
        guidelines, meaning that a drug-event pair is considered positive with 
        the ROR approach when an ROR lower-bound value equal to or larger than 
        1 (bound_threshold) and number of occurrences equal to or larger than 3
        (pair_threshold). Then, these are compared to the labels assigned to
        each drug-reaction pair in the input data. For area under the curve 
        calculations, the ROR lower-bound values are normed to [0,1] and pairs 
        with 0 occurrences are manually set to 0. The method operates on an 
        input pandas dataframe that has the same structure as the main_df, 
        which can be used to provide only a slice view instead of the entire 
        dataframe. The input dataframe can be further filtered with a list of 
        lists (partial) that has [drug, reaction, ...] structure, intended to 
        be used with the list of negative sampled train-test data or the 
        reference sets. This method also accepts a d_type argument (e.g.: 
        train, test) which is used in the filename generation for the
        resulting dictionary to be written out into, see ror_results_dict_train
        and ror_results_dict_test for further details.
        
    prep_reference_set():
        This method is launched by the process_files() method when tt_split = 0
        and a reference dataset is provided. Then, it processes the 
        reference_dict into a list of lists [drug, reaction, label] by dropping
        entries that lack drug or reaction features.
        
    prep_out_csv():
        Initiates the shared CSV output (when out_csv is specified) by printing
        a header for it using predetermined values and the grid_params 
        dictionary. This step will create multiple header lines in the CSV file
        as each separate run will print one, but that can be used to 
        distinguish the runs, given they were done sequentially. This method is
        also responsible for writing out the calculated ROR metrics, when
        calc_ror_performance is true.
                
    get_logger():
        Creates and returns a tf.keras.Model() compatible CSVLogger callback 
        object, that is used to print out the loss values of the model training
        at the end of each epoch into a CSV file inside the output folder.
        
    get_out_dir():
        Returns the class variable out_dir, which is the full path of the
        generated output folder. Intended to be called outside of the class,
        so the folder can be accessed by other methods.
   

    """
    
    def __init__(self, input_settings):
        self.input_settings = input_settings
        self.read_settings()
        
        
    def read_settings(self):
        print(str(datetime.now()) + ' - Reading input parameters')
        
        # Source files
        self.main_data_path = self.input_settings["oSourceList"]["oMainData"]
        self.reaction_embedding_path = self.input_settings["oSourceList"]["oReactionEmbedding"]
        self.drug_features_chem_path = self.input_settings["oSourceList"]["oDrugFeaturesChem"]
        self.drug_features_bio_path = self.input_settings["oSourceList"]["oDrugFeaturesBio"]
        self.reference_data_path = self.input_settings["oSourceList"]["oReferenceDatasets"]
        self.pretrained_model_path = self.input_settings["oSourceList"]["oPretrainedModel"]
        
        # General parameters
        self.calc_ror_performance = self.input_settings["oParameterList"]["oCalculateRorPerformance"]
        
        # Data processing parameters
        self.no_emb = self.input_settings["oParameterList"]["oNoEmbedding"]
        self.negative_sample = self.input_settings["oParameterList"]["oNegativeSampling"]["Enabled"]
        self.negsamp_by_type = self.input_settings["oParameterList"]["oNegativeSampling"]["oByData"].upper()
        self.negsamp_use_freq = self.input_settings["oParameterList"]["oNegativeSampling"]["oUseFreqSampling"]
        self.repeat_sampling = self.input_settings["oParameterList"]["oNegativeSampling"]["oRepeatSampling"]
        self.standardize_reacts = self.input_settings["oParameterList"]["oStandardizeReactions"]
        self.standardize_drugs = self.input_settings["oParameterList"]["oStandardizeDrugs"]
        self.tt_split = self.input_settings["oParameterList"]["oTrainTestSplit"]
        
        # Training parameters
        self.optimizer = self.input_settings["oParameterList"]["oOptimizer"]
        self.learning_rate = self.input_settings["oParameterList"]["oLearningRate"]
        self.numof_epochs = self.input_settings["oParameterList"]["oEpochs"]
        self.batch_size = self.input_settings["oParameterList"]["oBatchSize"]
        
        # Output path
        self.output_path = self.input_settings["oOutputPath"]
        self.project_name = self.input_settings["oProjectName"].lower()
        
        self.out_dir = io.make_out_dir(self.output_path, self.project_name)
        self.out_csv = self.input_settings["oOutCsv"]
        
        # Parameters for shared CSV file during multiple runs (e.g. grid search)
        self.grid_params = {
            "embedding_file": self.reaction_embedding_path,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "negsamp_bydata": self.negsamp_by_type,
            "negsamp_freq": self.negsamp_use_freq,
            "negsamp_rep": self.repeat_sampling,
            "location": self.out_dir
            }


    def train(self):
        
        self.read_files()
        self.process_files()
        self.build_model()
        self.train_model()
        self.test_and_save()
        
        
    def read_files(self):
        print(str(datetime.now()) + ' - Reading input files')
        
        self.main_df = pd.read_csv(self.main_data_path, delimiter = '|', 
                                   names = ['COMB_NAME', 'DRUG', 'REACTION', 'LABEL', 'FREQ', 'ROR'], header = None)
        
        self.drug_dict_chem = io.read_vector_dict(self.drug_features_chem_path, '|', False)
        self.drug_dict_bio = io.read_vector_dict(self.drug_features_bio_path, '|', False)
        self.reaction_dict = io.read_vector_dict(self.reaction_embedding_path, '|', False)
        self.reference_dict = {}
        
        # Read in the reference data files, when provided, as a dictionary for easy merging of duplicates
        for ref_file in self.reference_data_path:
            io.read_test_set(ref_file, '|', False, self.reference_dict)
                    
        
    def process_files(self):
        print(str(datetime.now()) + ' - Processing input files')
        
        # Filter out empty drug-feature entries (caused by faulty data preparation)
        for key in list(self.drug_dict_chem.keys()):
            if len(self.drug_dict_chem[key]) == 0:
                del self.drug_dict_chem[key]
                del self.drug_dict_bio[key]
                
        # Match drug features to pairs
        self.main_df['CHEM'] = self.main_df["DRUG"].map(self.drug_dict_chem)
        self.main_df['BIO'] = self.main_df["DRUG"].map(self.drug_dict_bio)
        
        # Change reaction-embedding dictionary to simple indices if no_emb mode is enabled
        if self.no_emb:
            self.reaction_dict = pp.swap_embedding_to_index(self.reaction_dict)
        
        # Match embedding features (or indices) to pairs
        self.main_df['EMB'] = self.main_df['REACTION'].map(self.reaction_dict)
        
        # Change labels of unvalidated reports from -1 (unlabeled) to 0 (negative class)
        # If no negative sampling is selected, they will be used as negative associations
        # If negative sampling is selected, they are discarded
        self.main_df['LABEL'] = self.main_df['LABEL'].apply(pd.to_numeric, errors='coerce')
        self.main_df['LABEL'] = np.where(self.main_df['LABEL'] == -1, 0, 1)
        
        # Prepare data for ROR performance calculations if enabled
        if self.calc_ror_performance:
            # Create a normalized (0-1 interval) ROR value for area under the curve calculations
            # Set the normalized ROR value to 0 when the number of reports (FREQ) is less than 3
            self.main_df['FREQ'] = self.main_df['FREQ'].apply(pd.to_numeric, errors='coerce')
            self.main_df['ROR'] = self.main_df['ROR'].apply(pd.to_numeric, errors='coerce')
            self.main_df['ROR_NORM'] = (self.main_df['ROR'] - self.main_df['ROR'].min()) / (self.main_df['ROR'].max() - self.main_df['ROR'].min())
            self.main_df['ROR_NORM'] = np.where(self.main_df['FREQ'] < 3, 0, self.main_df['ROR_NORM'])
    
            # Set FREQ, ROR and ROR_NORM to 0 when they are NaN for any reason
            self.main_df['FREQ'] = np.where(pd.isna(self.main_df['FREQ']), 0, self.main_df['FREQ'])
            self.main_df['ROR'] = np.where(pd.isna(self.main_df['ROR']), 0, self.main_df['ROR'])
            self.main_df['ROR_NORM'] = np.where(pd.isna(self.main_df['ROR_NORM']), 0, self.main_df['ROR_NORM'])
                    
        # Assamble test data from a reference set if enabled (by setting train-test split ratio to 0)
        if self.tt_split == 0 and len(self.reference_dict) != 0:
            self.reference_data = self.prep_reference_set()
            
            # Measure ROR performance on the reference set if enabled, before entries are dropped
            # Use the original reference dictionary, where entries without features are still there
            if self.calc_ror_performance:
                self.ror_results_dict_test = self.get_ror_performance(self.main_df, 'test', partial = list(self.reference_dict.values()))
            
            # Filter out the entries of the validated positive set that are present in the reference set too
            for index, row in self.main_df.iterrows():
                if row['COMB_NAME'] in self.reference_dict and row['LABEL'] == 1:   
                    self.main_df.drop(index, inplace = True)
                    
        elif self.tt_split == 0 and len(self.reference_dict) == 0:
            sys.exit("[Error] - train-test split of 0 means that a reference set will be used for testing but non was given.")
            
        else:
            self.reference_data = None
        
        original_len = len(self.main_df)

        print(str(datetime.now()) + ' - Number of unique reactions in main data: ' + str(len(self.main_df['REACTION'].unique())))
        print(str(datetime.now()) + ' - Number of unique drugs in main data: ' + str(len(self.main_df['DRUG'].unique())))
        print(str(datetime.now()) + ' - Number of unique pairs in main data: ' + str(original_len))
        print(str(datetime.now()) + ' - Number of positive pairs in main data: ' + str(len(self.main_df[self.main_df['LABEL'] == 1])))

        # Filter out drug-reaction pairs without drug or reaction features
        self.main_df = self.main_df.dropna(axis = 0)
        
        print(str(datetime.now()) + ' - Number of unique reactions after dropnan: ' + str(len(self.main_df['REACTION'].unique())))
        print(str(datetime.now()) + ' - Number of unique drugs after dropnan: ' + str(len(self.main_df['DRUG'].unique())))
        print(str(datetime.now()) + ' - Number of unique pairs after dropnan: ' + str(len(self.main_df)))
        print(str(datetime.now()) + ' - Number of positive pairs after dropnan: ' + str(len(self.main_df[self.main_df['LABEL'] == 1])))
        print(str(datetime.now()) + ' - Number of lost pairs: ' + str(original_len - len(self.main_df)))
        
        self.train_data, self.test_data, self.test_entries = self.prepare_data()
        
        
    def prepare_data(self):
        
        if self.negative_sample:
            print(str(datetime.now()) + ' - Sampling negative data from validated')
            
            if self.negsamp_by_type == "NONE":
                sampled_train, sampled_test, false_neg_rate = pp.negative_sample_train_test_split_full_random(
                    self.main_df, self.tt_split, self.negsamp_use_freq, self.repeat_sampling)
                
            elif self.negsamp_by_type == "DRUG" or self.negsamp_by_type == "REACTION":
                sampled_train, sampled_test, false_neg_rate = pp.negative_sample_train_test_split_by_type(
                    self.main_df, self.tt_split, self.negsamp_use_freq, self.negsamp_by_type, self.repeat_sampling)
            else:
                sys.exit("[Error] - valid data type parameters for negative sampling are: drug, reaction, none.")
            
            # Measure ROR performance on the test and train sets if enabled
            if self.calc_ror_performance:
                if self.reference_data == None:
                    self.ror_results_dict_test = self.get_ror_performance(self.main_df, 'test', partial = sampled_test)
                self.ror_results_dict_train = self.get_ror_performance(self.main_df, 'train', partial = sampled_train)
            
            # Check if reference data was successfully processed and if so, use it instead
            if self.reference_data != None:
                sampled_test = self.reference_data

            io.write_list_of_lists(self.out_dir, 'neg_sampled_train_data', sampled_train, '|')
            io.write_list_of_lists(self.out_dir, 'neg_sampled_test_data', sampled_test, '|')
            print(str(datetime.now()) + ' - Rate of falsely negative sampled pairs in test data: ' + str(false_neg_rate))
  
            # Match train and test data pairs with their corresponding features in numpy array formats
            train_emb_np = np.stack([self.reaction_dict[x[1]] for x in sampled_train])
            test_emb_np = np.stack([self.reaction_dict[x[1]] for x in sampled_test])
                
            train_chem_np = np.stack([self.drug_dict_chem[x[0]] for x in sampled_train])
            test_chem_np = np.stack([self.drug_dict_chem[x[0]] for x in sampled_test])
            
            train_bio_np = np.stack([self.drug_dict_bio[x[0]] for x in sampled_train])
            test_bio_np = np.stack([self.drug_dict_bio[x[0]] for x in sampled_test])
                
        else:
            # Check if reference data was successfully processed or train-test split needs to be done
            if self.reference_data != None:
                train_split = self.main_df
                sampled_test = self.reference_data
                
                # Measure ROR performance on the train set if enabled
                if self.calc_ror_performance:
                    self.ror_results_dict_train = self.get_ror_performance(train_split, 'train')
                
                # Match test data pairs with their corresponding features in numpy array formats
                test_emb_np = np.stack([self.reaction_dict[x[1]] for x in sampled_test])
                test_chem_np = np.stack([self.drug_dict_chem[x[0]] for x in sampled_test])
                test_bio_np = np.stack([self.drug_dict_bio[x[0]] for x in sampled_test])
                
            else:
                train_split, test_split = train_test_split(self.main_df, test_size = self.tt_split)
                
                # Match train and test data pairs with their corresponding features in numpy array formats
                test_emb_np = np.stack(test_split['EMB'].values)
                test_chem_np = np.stack(test_split['CHEM'].values)
                test_bio_np = np.stack(test_split['BIO'].values)
                
                # Measure ROR performance on the test and train sets if enabled
                if self.calc_ror_performance:
                    self.ror_results_dict_test = self.get_ror_performance(test_split, 'test')
                    self.ror_results_dict_train = self.get_ror_performance(train_split, 'train')
                
                # Create list of test entries that otherwise here would not be created
                sampled_test = test_split[['DRUG', 'REACTION', 'LABEL']].values.tolist()

            # Match train data pairs with their corresponding features in numpy array formats
            train_emb_np = np.stack(train_split['EMB'].values)
            train_chem_np = np.stack(train_split['CHEM'].values)
            train_bio_np = np.stack(train_split['BIO'].values)
        
        self.chem_len = train_chem_np.shape[1]
        self.bio_len = train_bio_np.shape[1]
        self.emb_len = train_emb_np.shape[1]
        
        print(str(datetime.now()) + ' - Standardizing features')
        # Calculate standardization scaler for the train data, apply it to both train and test data
        
        scaler = ""
        
        if self.standardize_reacts and self.standardize_drugs and not self.no_emb:
            scaler, scaled_train = pp.standardize_simple_np(np.concatenate((train_emb_np, train_chem_np), axis = 1))
            train_data = np.concatenate((scaled_train, train_bio_np), axis = 1)
            
            scaled_test = scaler.transform(np.concatenate((test_emb_np, test_chem_np), axis = 1))
            test_data = np.concatenate((scaled_test, test_bio_np), axis = 1)
            
        elif self.standardize_reacts and not self.no_emb:
            scaler, scaled_train = pp.standardize_simple_np(train_emb_np)
            train_data = np.concatenate((scaled_train, train_chem_np, train_bio_np), axis = 1)
            
            scaled_test = scaler.transform(test_emb_np)
            test_data = np.concatenate((scaled_test, test_chem_np, test_bio_np), axis = 1)
            
        elif self.standardize_drugs:
            scaler, scaled_train = pp.standardize_simple_np(train_chem_np)
            train_data = np.concatenate((train_emb_np, scaled_train, train_bio_np), axis = 1)
            
            scaled_test = scaler.transform(test_chem_np)
            test_data = np.concatenate((test_emb_np, scaled_test, test_bio_np), axis = 1)
            
        else:
            train_data = np.concatenate((train_emb_np, train_chem_np, train_bio_np), axis = 1)
            test_data = np.concatenate((test_emb_np, test_chem_np, test_bio_np), axis = 1)
            
        if scaler != "":
            dump(scaler, self.out_dir + 'std_scaler.bin', compress = True)
            
        # Match final datasets with their corresponding truth labels                
        if self.negative_sample:
            train_data = np.append(train_data, np.asarray([x[2] for x in sampled_train]).reshape((-1, 1)), axis = 1)
            test_data = np.append(test_data, np.asarray([x[2] for x in sampled_test]).reshape((-1, 1)), axis = 1)
        else:
            train_data = np.append(train_data, np.asarray(train_split['LABEL']).reshape((-1, 1)), axis = 1)
            if self.tt_split != 0:
                test_data = np.append(test_data, np.asarray(test_split['LABEL']).reshape((-1, 1)), axis = 1)
            else:
                test_data = np.append(test_data, np.asarray([x[2] for x in sampled_test]).reshape((-1, 1)), axis = 1)
            
        return train_data, test_data, sampled_test
    
    
    def build_model(self):
        print(str(datetime.now()) + ' - Building model')
        
        if self.pretrained_model_path != "":
            # Read a previously trained model when a location is specified
            self.model = load_model(self.pretrained_model_path)
        else:
            if self.no_emb:
                self.model = models.build_classifier_without_emb(
                            self.chem_len, self.bio_len, self.emb_len, len(self.reaction_dict))
            else:
                self.model = models.build_classifier(
                            self.chem_len, self.bio_len, self.emb_len)
    
            self.model.compile(loss = 'binary_crossentropy', optimizer = self.get_optimizer(), 
                          metrics = ['accuracy', tf.keras.metrics.AUC(name = 'auroc'), 
                                                  tf.keras.metrics.AUC(curve = 'PR', name = 'auprc')])
            
        self.visualize_model()
        
        
    def train_model(self):
        
        csv_logger = CSVLogger(self.out_dir + 'loss.csv', append = True, separator = '|')
        
        if self.out_csv != "":
            # When a location is given, prepare a shared CSV for the results 
            # that can be written by multiple instances of the classifier
            self.prep_out_csv()
        
        for epoch in range(self.numof_epochs):
            print(str(datetime.now()) + " - Commencing training of epoch {}".format(epoch + 1))
            
            np.random.shuffle(self.train_data)
                
            self.model.fit(
                [self.train_data[:, 0:self.chem_len], self.train_data[:, self.chem_len:self.chem_len + self.bio_len], 
                 self.train_data[:, self.chem_len + self.bio_len:-1]],
                 self.train_data[:, -1], batch_size = self.batch_size, shuffle = False,  callbacks = [csv_logger])
            
            # After every 10th epoch, run a test with either the split-test data or the external reference set
            if ((epoch + 1) % 10 == 0):
                
                eval_results = self.model.evaluate(
                    [self.test_data[:, 0:self.chem_len], self.test_data[:, self.chem_len:self.chem_len + self.bio_len], 
                      self.test_data[:, self.chem_len + self.bio_len:-1]], \
                      self.test_data[:, -1], verbose = 0)
                
                print(str(datetime.now()) + ' - test_loss=%.4f, test_accuracy=%.4f, test_auroc=%.4f, test_auprc=%.4f' %(
                    eval_results[0], eval_results[1], eval_results[2], eval_results[3]))
                
                if (self.out_csv != "" and (epoch + 1) % 50 == 0):
                    next_row = [eval_results[0], eval_results[1], eval_results[2], eval_results[3], epoch + 1]
                    next_row.extend(list(self.grid_params.values()))
                    io.write_by_append(self.out_csv, next_row)
        
        
    def test_and_save(self):
        # Run a prediction on the test set, write the results into a CSV and save the model
        
        test_preds = self.model.predict([self.test_data[:, 0:self.chem_len], self.test_data[:, self.chem_len:self.chem_len + self.bio_len], 
                                         self.test_data[:, self.chem_len + self.bio_len:-1]])
        test_preds_with_names = []
        
        for i in range(len(test_preds)):
            entry = self.test_entries[i]
            entry.extend(test_preds[i])
            test_preds_with_names.append(entry)
            
        io.write_list_of_lists(self.out_dir, 'test_data_with_predictions', test_preds_with_names, '|')
        self.model.save(self.out_dir + 'model.h5')

        
    def visualize_model(self):
        print(self.model.summary())
        tf.keras.utils.plot_model(self.model, to_file = self.out_dir + 'model.png',
                              show_shapes = True, show_layer_names = True, rankdir = 'TB', 
                              expand_nested = False, dpi = 96)
        
        
    def get_optimizer(self):
        # Legacy versions are needed for graph mode, as of Tensorflow 2.11
        
        if self.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate = self.learning_rate, 
                                                     rho = 0.9, 
                                                     momentum = 0.0, 
                                                     epsilon = 1e-07, 
                                                     centered = False, 
                                                     name = 'RMSprop')
        elif self.optimizer == 'adagrad':
            optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(
                                                    learning_rate = self.learning_rate,
                                                    initial_accumulator_value = 0.1,
                                                    l1_regularization_strength = 0.0,
                                                    l2_regularization_strength = 0.0,
                                                    use_locking = False,
                                                    name = 'ProximalAdagrad')

        elif self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.legacy.Adam(
                                                    learning_rate = self.learning_rate,
                                                    beta_1 = 0.9,
                                                    beta_2 = 0.999,
                                                    epsilon = 1e-07,
                                                    amsgrad = True,
                                                    name = 'Adam')

        elif self.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.legacy.SGD(
                                                    learning_rate = self.learning_rate, 
                                                    momentum = 0.9, 
                                                    nesterov = True, 
                                                    name = 'SGD')
        else:
            sys.exit("[Error] - valid optimizer parameters are: rmsprop, adagrad, adam, sgd.")
            
        return optimizer
         
    
    def get_ror_performance(self, input_df, d_type, pair_threshold = 3, bound_threshold = 1, partial = None):
        # Measure the performance of the pre-calculated ROR lower-bound values
        # using the accepted criteria for positive association
        # when there at least 3 separate reports of them
        # and the ROR lower-bound is at least 1
        print(str(datetime.now()) + ' - Calculating ROR for ' + d_type + ' data')
        coverage = -1
            
        if partial is not None:
            # Filter the main dataframe for the entries in 'partial' when specified
            keys = [x[0] + '_' + x[1] for x in partial]
            data = input_df[input_df['COMB_NAME'].isin(keys)].copy()
            data.drop(columns=['EMB', 'CHEM', 'BIO'], axis = 1, inplace = True)
            coverage = data.shape[0] / len(partial)
            
            # Create a helper set (pairs_in_data) for faster searching of entries
            pairs_in_data = set(data.COMB_NAME.unique())
            pair_to_label = {}
            
            for entry in partial:
                # Add entries from partial that are not present in the main dataframe
                # Set their values to zeros (always negative class)
                # Create a helper dictionary (pair_to_label) for faster label matching later
                key = entry[0] + '_' + entry[1]
                pair_to_label[key] = entry[2]
                if key not in pairs_in_data:
                    new_row = {'COMB_NAME': key, 'DRUG': entry[0], 'REACTION': entry[1], 
                               'LABEL': entry[2], 'FREQ': 0, 'ROR': 0, 'ROR_NORM': 0}
                    data.loc[len(data)] = new_row

            # Match the corresponding labels for the rest of the entries
            data['LABEL'] = data["COMB_NAME"].map(pair_to_label)                
            data.to_csv(self.out_dir + 'ror_data_' + d_type + '.csv', index = False)
            
        else:
            data = input_df
                    
        tt = len(data[((data['FREQ'] >= pair_threshold) 
                      & (data['ROR'] >= bound_threshold)) & (data['LABEL'] == 1)]) #True Positive
        ft = len(data[((data['FREQ'] < pair_threshold) 
                      | (data['ROR'] < bound_threshold)) & (data['LABEL'] == 1)]) #False Negative
        tf = len(data[((data['FREQ'] >= pair_threshold) 
                      & (data['ROR'] >= bound_threshold)) & (data['LABEL'] == 0)]) #False Positive
        ff = len(data[((data['FREQ'] < pair_threshold) 
                      | (data['ROR'] < bound_threshold)) & (data['LABEL'] == 0)]) #True Negative
        al = len(data)
        
        # Accuracy
        if al != 0:
            acc = (tt + ff) / al
        else:
            acc = -1
        
        # Precision
        if (tt + tf) != 0:
            prec = tt / (tt + tf)
        else:
            prec = -1
        
        # Recall or sensitivity
        if (tt + ft) != 0:
            recall = tt / (tt + ft)
        else:
            recall = -1
        
        # Specificity
        if (ff + tf) != 0:
            spec = ff / (ff + tf)
        else:
            spec = -1
            
        if len(data['LABEL']) != 0:
            auroc = roc_auc_score(data['LABEL'], data['ROR_NORM'])
            auprec, aurecall, _ = precision_recall_curve(data['LABEL'], data['ROR_NORM'])
            auprc = auc(aurecall, auprec)
        else:
            auroc = -1
            auprc = -1
            
        results = {'Accuracy': acc, 'Precision': prec, 'Recall': recall,
                'Specificity': spec, 'AUROC': auroc, 'AUPRC': auprc, 
                'Test_coverage': coverage}
        
        print(results)
        io.write_simple_dict(self.out_dir, 'ror_results_' + d_type, results, '|')
                
        return results
    
    
    def prep_reference_set(self):
        
        reference_list = []
        
        # Filters out entries without drug or reaction features
        for value in self.reference_dict.values():
            if value[0] in self.drug_dict_chem and value[1] in self.reaction_dict:
                entry = [value[0], value[1], value[2]]
                reference_list.append(entry)
                
        return reference_list
      
    
    def prep_out_csv(self):
        
        first_row = ['loss', 'accuracy', 'auroc', 'auprc', 'epoch']
        first_row.extend(list(self.grid_params.keys()))
        io.write_by_append(self.out_csv, first_row)
        
        if self.calc_ror_performance:
            ror_row_test = ['ROR_test', self.ror_results_dict_test['Accuracy'], self.ror_results_dict_test['AUROC'], 
                       self.ror_results_dict_test['AUPRC'], self.ror_results_dict_test['Test_coverage']]
            ror_row_test.extend(list(self.grid_params.values()))
            io.write_by_append(self.out_csv, ror_row_test)
            
            ror_row_train = ['ROR_train', self.ror_results_dict_train['Accuracy'], self.ror_results_dict_train['AUROC'], 
                       self.ror_results_dict_train['AUPRC'], self.ror_results_dict_train['Test_coverage']]
            ror_row_train.extend(list(self.grid_params.values()))
            io.write_by_append(self.out_csv, ror_row_train)
        
        
    def get_logger(self):
        return CSVLogger(self.out_dir + 'loss.csv', append = True, separator = '|')
        
        
    def get_out_dir(self):
        return self.out_dir


#%% Initiate main sequence

if __name__ == "__main__":
    
    # Start timer for computation time measurements
    start_time = time.time()
    # Read the provided json file from command-line arguments
    input_settings = io.read_json_input(sys.argv[1])
    # Initialize the main class
    classifier = Classifier(input_settings)
    # Initialization of the class creates the output directory, where the input json file can now be copied
    io.copy_to_out_dir(classifier.get_out_dir(), sys.argv[1], 'input.json')
    # Redirect print() and errors so that it will also write into a log.txt at out_dir
    sys.stdout = io.Logger(sys.stdout, classifier.get_out_dir())
    sys.stderr = sys.stdout
    # Begin model training, which includes data reading and preprocessing
    classifier.train()
    # Conclude
    print(str(datetime.now()) + ' - All done!')
    elapsed_time = time.time() - start_time
    print(str(datetime.now()) + " - Only took %02d seconds!" % (elapsed_time))

