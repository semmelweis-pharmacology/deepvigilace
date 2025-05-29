#%% Import libraries

import sys
import time

import numpy as np
import multiprocessing as mp
import itertools as it

from datetime import datetime
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Project specific modules
from lib import io_file_handling as io
from lib import preprocessing as pp
from lib import sim_metrics as sm
from lib import math_utils as mu


#%% Define main class

class SimAnalysis():
        
    """
    The SimAnalysis class encapsulates the statistical analysis of the 
    embedding vector spaces using various validated datasets, as well as 
    provides single query results for top closest neighbors evaluation.

    ...
    
    Args
    ---------
    input_settings: json object from the input file, using json.load()
        The input JSON file defines the location of the necessary source files, 
        the path to the output directory and the changeable hyperparameters for
        the training.  See the template JSON and the class attributes for 
        further details.

    Attributes
    ----------
    input_settings: json object, see in Args
    
    embedding_path: str
        Full path to the main input data file (CSV, delimiter '|', no header), 
        containing in the first column the MedDRA PT name of the embedded 
        adverse event reactions, while other columns are populated by the
        elements of the corresponding embedding vectors, containing one 
        element (float) per cell. Consequently, this file is the product of
        the embedding module.
        
    meddra_hierarchy_path: str
        Full path to the MedDRA hierarchy dictionary file (CSV, delimiter '|'),
        which contains the merger of all MedDRA terminology levels (LLT, PT, 
        HLT, HLGT, SOC). In each row there is a unique line, describing a 
        specific "path" the given reaction takes to reach an SOC category 
        (highest), meaning that lines belonging to the same reaction may only 
        differ in as few as only one column. To each unique PT name belongs 
        one "primary" path and might belong several "secondary" ones (denoted 
        in the last column by "rank"). Depending on the chosen metrics, all or 
        only the primary ones are considered for the calculations. The main 
        purpose of this file is to provide the MedDRA hierarchy information for
        the MedDRA-based hierarchical semantic similarity calculations.
        
        Disclaimer: as our MedDRA file is not provided due to licensing, these 
        calculations cannot be completed. When no MedDRA and or SMQ files are 
        given, all MedDRA, SMQ and path related metrics should be set to False 
        in the input JSON, which will prevent any crashes of the analyser. 
        See hier_semants for further details.
        
    smq_hierarchy_path: str
        Same as with MedDRA categories. For further deteals, see 
        meddra_hierarchy_path. Only difference is, the hierarchies are rather
        sparse, meaning that several reactions lack lower level (SMQ3-5)
        categories. The file is not provided either, as SMQ is part of MedDRA.
        
    reaction_freq_path: str
        Full path to the target term frequency file (CSV), containing a two-
        columned list (delimiter '|', no header) of reaction-freaquency pairs.
        The frequency, or more precisely, the number of occurrences (int) of 
        each reaction was calculated based on the entire report corpus,
        before rare terms under a threshold (defined during preprocessing, 11 
        in our case) were removed. The main purpose of this file is to 
        provide the frequency information for the infomration content (IC) 
        based calculations (MedDRA and SMQ only).
        
    validated_pairs_path: list of str
        A list of full paths to the validated drug-ADR dataset files (CSV,
        delimiter '|'). These contain three columns (DRUG, REACTION, TRUTH), 
        where the first is the DrugBank name, followed by the MedDRA PT and a 
        "1", which indicates these pairs all showed positive associations in 
        their respective studies. The truth value is not used in this module of
        the project. The content of the files are merged so that duplicates are
        eliminated and reactions belonging to the same drug in different files 
        are groupped together. As some reactions are excluded from the 
        embedding part (due to being too rare), it is possible that some 
        entries in these files will not be used either.
        
    ime_reactions_path: str
        Full path to an alternative MedDRA hierarchy file (CSV, delimiter '|')
        which contains only the IME (important medical events) reactions,
        defined by EMA, in a simplified form. The file is produced by 
        extracting the entries without comments from the original, publicly
        available source (.xlsx) into a CSV file. It is only used, when 
        only_ime_reactions is set to True, to filter out avderse reactions 
        which are not included in the IME list. This essencially eliminates 
        many common reactions that might be less interesting, but the remaining
        reactions usually have very low frequencies in the original corpus data
        and so, their embedding vectors are expected to be of lower quality in
        case of using the NSG embedding model.
        
    complete_distance_matrices_path: str
        Full path to the directory (ending in '/') where pre-calculated 
        distance matrices can be found. These files (CSV, delimiter '|') are 
        produced by this module when calculate_all and write_out_all are both 
        set to True, which is a costly process in both time and storage 
        requirements, but enables quick re-analysis by this module using 
        different parameters, as providing a proper 
        complete_distance_matrices_path will result in the matrices being read 
        in, instead of recalculated for validated data queries. See
        calculate_all and calculate_sims_for_valid for further details.
        
    metrics_list: list of str
        A list of strings extracted from the input JSON so that only the ones
        that are set to True will be included. The number of requested metrics
        greatly affects the computation time and many are either redundant or
        hard to interpret. Recommended: cosine, euclidean, dot, braycurtis,
        canberra.
    
    stats_list: list of str
        A list of strings extracted from the input JSON so that only the ones
        that are set to True will be included. The number of requested 
        statistical tests greatly affects the computation time and many are 
        either redundant or non relevant. Recommended: kolsir, meanmwhitney,
        varbrown, permdispair, permanova.

    calculate_all: boolean
        When set to True, the entire distance matrices are calculated, which
        is a time consuming process. When write_out_all is also set to True,
        these matrices are written into CSV files. Single queries do not use
        these matrices nor the files, only validated queries do. Consequently,
        when calculate_all is set to False, but calculate_sims_for_valid is set
        to True, the distance matrices are calculated (or read in, depending
        on whether complete_distance_matrices_path is empty or not) either way.
    
    write_out_all: boolean
        Determines whether the full distance matrices are written into CSV 
        files or not. See calculate_all and complete_distance_matrices_path for
        further details.
    
    single_query: str
        When not empty, this module will attempt to match the given reaction
        PT name to the ones available. When failed, the program quits. When the
        reaction is successfully identified, all distances/similarities set to
        True in the input JSON will be calculated for this reaction against all
        other reactions in the data. The results are then sorted and the top 25 
        closest/most similar entries for each metric are written out to the
        console as well as into a CSV file (delimiter '|'). Leave this 
        parameter empty to disable the single query calculation. Not case 
        sensitive.
    
    calculate_sims_for_valid: boolean
        When set to True, the module calculates a validated query, which is
        the most time-consuming part of this module. During its run, the 
        validated data files from validated_pairs_path are gathered and merged,
        so that each drug has a number of unique reactions associated with 
        it. These reaction groups will represent the "valid" group for their
        corresponding drug. For each valid group, a "random" group of 
        equivalent size is then created by uniform random sampling in such a 
        way, that there are no reactions belonging to both groups for the same 
        drug. The previously calculated distance matrices are used to select
        the corresponding distances for each group, then statistical tests
        are performed, comparing the distances in each valid group with the
        the ones in the corresponding random group. Given significant 
        differences between the valid data and the random (noise), we can draw
        conclusions whether the embedding space shows any real life relevance
        and organization.
        
    rand_sample_from_groupping: boolean
        When set to True, this parameter changes the validated query in a way, 
        that the uniform random sampled reactions for the "random" groups are
        restricted to the ones present in the files listed in 
        validated_pairs_path. This eliminates many noisy reactions (less known,
        less frequent) and so, reduces the statistical difference of the valid
        and random groups.
        
    normalize_vectors: boolean
        When set to True, the term vectors are first normalized (reduced to
        unit length). This essentially makes the cosine (normed dot) and the
        dot calculations equal. Not recommended for use as it is not very 
        practical, only serves testing purposes.
        
    only_ime_reactions: boolean
        When set to True, only the term vectors of IME reactions will be
        analysed. See ime_reactions_path for further details.
        
    filter_by_freq: boolean
        When set to True, only the term vectors with a certain number of 
        occurrences (or higher) will be analysed. This is an effective way of
        getting rid of the noisy vectors when the NSG model is used, which
        reduces the statistical differences between the random and valid groups
        during validated queries. Similarly to normalize_vectors, it only 
        serves testing purposes and not recommended to use.
        
    filter_threshold: int
        The threshold for the filtering of low-frequency reactions. See 
        filter_by_freq for further details. As the frequency distribution
        follows an exponential curve, numbers around 1000 eliminate ~78% of the
        terms already, even though the most frequent ones go up to millions.
        
    stat_perm_num: int
        The number of permutations for the PERMANOVA and PERMDISP-pairwise
        statistical tests, performed on the valid and random reaction groups 
        during validated queries. Higher numbers pose a significant time cost 
        to the calculations, and are generally not worth it. Recommended: 999 
        (due to the way the p-values are calculated, this will mean 1000 
        different tests to consider and a nicer decimal value for p). This 
        number is ignored when the number of possible permutations for a test
        is lower. In that case, all permutations are used.
 
    output_path: str
        Full path to the output directory (ending in '/'), where a new folder 
        will be created, named by the project_name + a time stamp.
        
    project_name: str
        Used for the creation of the output folder, can be empty.
    
    out_dir: str
        The output folder used to print out the result files. See project_name
        and output_path for further details.
        
    embeddings: dictionary
        Contains the MedDRA PT name of reactions as keys and the corresponding
        term vectors  (numpy arrays, float64) as values. This serves as the
        main input data for the similarity analysis module.
        
    meddra_sep_hier_dict: dictionary
        Contains the MedDRA PT name of reactions as keys and a list of their
        corresponding unique MedDRA "paths" (see meddra_hierarchy_path) in the
        form of dictionaries for easier access, in which the keys are the
        different MedDRA levels (PT, HLT, HLGT, SOC) and whether it is primary
        or secondary (RANK) and values are the name of the level.
        
    smq_sep_hier_dict: dictionary
        Contains the MedDRA PT name of reactions as keys and a list of their
        corresponding unique SMQ "paths" (see smq_hierarchy_path) in the form 
        of dictionaries for easier access, in which the keys are the different 
        SMQ levels (SMQ1-5) and values are the name of the level.
        
    validated_dict: dictionary
        Contains the DrugBank name of drugs from the validated drug-ADR
        datasets (see validated_pairs_path) as keys, and a list of unique
        associated reaction MedDRA PTs as values. All datasets are merged 
        together. This greatly influences the validated queries, as it serves
        as the basis for its calculations.
        
    freq_dict: dictionary
        Contains the MedDRA PT name of reactions as keys and their number of 
        occurrences in the corpus as values. See reaction_freq_path for further
        details.
        
    hier_semants: bool
        A supporting variable that is used to check whether any MedDRA-based or
        SMQ-based measurements are set to True in the input JSON. When all are
        False, MedDRA and SMQ files are not read in. This is useful when user
        is in no posession of said files. See meddra_hierarchy_path for further
        details.
        
    ic_meddra: dictionary
        Contains the MedDRA PT name of reactions along with all their 
        associated higher level categories (HLT, HLGT, SOC) as keys and the 
        calculated information content of the given term as values.
    
    ic_smq: dictionary
        Contains the MedDRA PT name of reactions along with all their 
        associated higher level SMQ categories (SMQ1-5) as keys and the 
        calculated information content of the given term as values.
        
    data_dict_all_dists: dictionary
        Contains the name of the calculated metrics as keys and the 
        corresponding distance matrices (ndarray) as values. The diagonal of
        the matrices is set to -infinity for easier exclusion, similarly to
        any unobtainable distance values due to missing data (mostly with 
        "metrics" using the SMQ hierarchy). This is either produced by an
        all-on-all query or read in from CSV files containing the precomputed
        matrices. See calculate_all and write_out_all for further details.
        
    props_dict_all_dists: dictionary
        Contains the name of the calculated metrics as keys and a simple
        properties dictionary for each as values. These property dictionaries
        contain precomputed global properties for the matrices, such as
        maximum, minimum, average and normed average. They serve the easier
        calculation of normed values during validated queries.
        
    group_distmats: dictionary
        Contains the DrugBank name of drugs from the validated drug-ADR
        datasets (see validated_pairs_path) as keys, and a dictionary of 
        metrics as values. These metric dictionaries contain the calculated
        metrics as keys and the corresponding distance matrices (ndarray) as
        values, but the matrices are sliced out of the complete ones in such a
        way that they only hold the values for the reactions of the valid and
        random groups of the given drug. See calculate_sims_for_valid for
        further details.
        
    valid_mean_sim: dictionary
        Contains the DrugBank name of drugs from the validated drug-ADR
        datasets (see validated_pairs_path) as keys, and a dictionary of 
        statistical tests and other results as values. These result 
        dictionaries contain the name of the calculated statistical tests 
        (e.g.: Mann- Whitney for means, PERMANOVA) or other results (means of 
        groups, gain between groups) from validated queries as keys with the 
        resulting values themselves as the values. These results are written 
        out in both a common and separate CSV files, as well as they are 
        aggregated in a summary file.
        

    Methods
    ----------
    read_settings():
        Extracts the input parameters from the provided json object into class
        variables. Furthermore, creates the output folder at the designated
        destination. Executed during class initialization. See out_dir and the 
        rest of the parameter descriptions for further details.
        
    query():
        The main method of the SimAnalysis class, intended to start the entire
        process with all the selected queries.
        
    read_files():
        Reads the input data files provided into memory, see embedding_path,
        meddra_hierarchy_path, smq_hierarchy_path, validated_pairs_path and 
        reaction_freq_path for further details.
        
    prepare_dictionaries():
        Performs light preprocessing steps, when the given parameters specify,
        such as using only IME reactions (see only_ime_reactions), normalizing
        the embedding vectors (see normalize_vectors) or filtering them by the
        frequency of the corresponding reaction (see filter_by_freq). It also
        calculates information content in advance, for each reaction, MedDRA 
        and SMQ terms, see hier_semants, ic_meddra and ic_smq for further 
        details.
        
    calculate_single():
        Executes a single query, where all the metrics from metrics_list are
        calculated between the so-called center reaction, defined in the input
        parameter (single_query), and all the other reactions in the data. The
        results are sorted so that the top 25 most similar/closest reactions
        are gathered for each metric and written to the console as well as into
        an output CSV file. For further details, see single_query.
        
    all_query():
        Executes an all-on-all query, where all the metrics from metrics_list 
        are calculated between all reactions in the data, resulting in large
        distance matrices, which are written out, when write_out_all is set to
        True, and are later used, when calculate_sim_for_valid is set to True.
        When complete_distance_matrices_path is not empty, this function will 
        instead attempt to read in the pre-calculated distance matrices from 
        the specified path. For further details, see calculate_all and 
        write_out_all.
        
    valid_query():
        Executes a validated query, where the validated drug-ADR pairs
        are used to create a valid and a random reaction group for each drug
        in the data. Then, distances are extracted for the two groups from the
        complete distance matrices for each metrics from metrics_list, and the
        values are statistically analyzed. For further details, see stats_list
        and calculate_sims_for_valid.
        
    para_process_distmats_for_valid():
        Initiates a parallelized block of the code by launching a 
        multiprocessing pool. Each process in this pool is responsible for
        performing the statistical analysis between the valid and random
        reaction group of a single drug. Returns valid_mean_sim.
        
    postprocessing():
        Performs postprocessing on valid_mean_sim by aggregating the results
        via either averaging or percentile of significant tests. Also writes
        out several result CSV files (one with all results, one for each result
        separately, one summary).        
        
    get_similarities_single_input():
        Initiates parallelized blocks of the code by lanching a multiprocessing
        pool for each metrics from metrics_list. Each process in this pool is
        responsible of calculating the current metric between the center
        reaction, defined in the input parameter (single_query), and the 
        current reaction in the loop. Returns a dictionary of dictionaries,
        where for each metric, as keys, belongs a dictionary of reaction-value
        pairs. The self-distances are later eliminated.
        
    para_process_distmats_for_all(metric):
        Initiates a parallelized block of the code by launching a 
        multiprocessing pool for the given metric (argument), while running in
        a loop, inside all_query(). Each process in this pool is responsible of
        calculating the current metric between a single reaction and all the 
        other reactions in the data. Returns a dictionary containing reaction 
        names as keys and a list of calculated metric results as the values. 
        This dictionary is later transformed into a proper distance matrix, and
        for each metric, collected in data_dict_all_dists.
        
    get_out_dir():
        Returns the class variable out_dir, which is the full path of the
        generated output folder. Intended to be called outside of the class,
        so the folder can be accessed by other methods.
    

    """
    
    def __init__(self, input_settings):
        self.input_settings = input_settings
        self.read_settings()
        
    
    def read_settings(self):
        
        # Source files
        self.embedding_path = self.input_settings["oSourceList"]["oEmbeddingDict"]
        self.meddra_hierarchy_path = self.input_settings["oSourceList"]["oMeddraDictionary"]
        self.smq_hierarchy_path = self.input_settings["oSourceList"]["oSmqDictionary"]
        self.reaction_freq_path = self.input_settings["oSourceList"]["oPtFreqDictionary"]
        self.validated_pairs_path = self.input_settings["oSourceList"]["oValidatedDictionary"]
        self.ime_reactions_path = self.input_settings["oSourceList"]["oImeReactionsList"]
        self.complete_distance_matrices_path = self.input_settings["oSourceList"]["oCompDistMatrices"]
        
        # List of metrics and stat test that will be analysed
        metrics_dict = self.input_settings["oParameterList"]["oMetrics"]
        self.metrics_list = [m.lower()[1:] for m in metrics_dict.keys() if metrics_dict[m] == True]
        stats_dict = self.input_settings["oParameterList"]["oStatTests"]
        self.stats_list = [t.lower()[1:] for t in stats_dict.keys() if stats_dict[t] == True]

        # Analysis parameters
        self.calculate_all = self.input_settings["oParameterList"]["oCalculateAll"]
        self.write_out_all = self.input_settings["oParameterList"]["oWriteOutAll"]
        self.single_query = self.input_settings["oParameterList"]["oSingleQuery"].upper()
        self.calculate_sims_for_valid = self.input_settings["oParameterList"]["oCalculateSimsForValidated"]
        self.rand_sample_from_groupping = self.input_settings["oParameterList"]["oRandomSampleFromValidOnly"]
        self.normalize_vectors = self.input_settings["oParameterList"]["oNormalizeEmbeddings"]
        self.only_ime_reactions = self.input_settings["oParameterList"]["oImeReactionsOnly"]
        self.filter_by_freq = self.input_settings["oParameterList"]["oFilterByFrequency"]
        self.filter_threshold = self.input_settings["oParameterList"]["oFilterThreshold"]
        self.stat_perm_num = self.input_settings["oParameterList"]["oStatPermNum"]

        # Output path
        self.output_path = self.input_settings["oOutputPath"]
        self.project_name = self.input_settings["oProjectName"].lower()
        
        self.out_dir = io.make_out_dir(self.output_path, self.project_name)


    def query(self):
        
        self.read_files()
        self.prepare_dictionaries()
        
        if self.single_query != '':
            self.calculate_single()
            
        if self.calculate_all and not self.calculate_sims_for_valid:
            print(str(datetime.now()) + " - Initiating all-on-all query")
            self.all_query()            
            
        if self.calculate_sims_for_valid:
            if self.complete_distance_matrices_path != '':
                print(str(datetime.now()) + " - Initiating validated data query using the precalculated matrices from "
                      + self.complete_distance_matrices_path)
                
            else:
                print(str(datetime.now()) + " - Initiating all-on-all query before validated data query")
                    
            self.all_query()
            self.valid_query()
                

    def read_files(self):
        print(str(datetime.now()) + ' - Reading input files')
        
        self.embeddings = io.read_vector_dict(self.embedding_path, '|', False)
        self.validated_dict = io.read_valid_pair_dicts_merged(self.validated_pairs_path, '|', True)
        self.freq_dict = io.read_simple_dict(self.reaction_freq_path, '|', False)
        
        # Check if meddra-based or smq-based measurements are required by the input json,
        # and only read in the files if needed - so that these can be easily disabled
        # when no MedDRA and SMQ files are available
        self.hier_semants = False
        
        for metric in self.metrics_list:
            
            if ('meddra' in metric or 'smq' in metric or 'path' in metric):
                self.hier_semants = True
                
        if self.hier_semants:
            
            self.meddra_sep_hier_dict = io.read_meddra_hierarchy_separate(self.meddra_hierarchy_path, '|', True, False)
            self.smq_sep_hier_dict = io.read_5level_hierarchy_separate(self.smq_hierarchy_path, '|', True, 'SMQ')
        
        
    def prepare_dictionaries(self):
        print(str(datetime.now()) + ' - Preparing dictionaries')
        
        if self.hier_semants:
            # Create dictionaries for information content using reaction frequencies
            self.ic_meddra = pp.process_freq_for_ic_meddra(self.freq_dict, self.meddra_sep_hier_dict)
            self.ic_smq = pp.process_freq_for_ic_smq(self.freq_dict, self.smq_sep_hier_dict)
        
        # Delete non-IME reactions if required
        if self.only_ime_reactions:
            
            ime_dict = io.read_simple_dict(self.ime_reactions_path, '\t', True)
            ime_dict2 = {value.upper(): key for key, value in ime_dict.items()}
            
            for k in list(self.embeddings.keys()):
                if k not in ime_dict2:
                    del self.embeddings[k]
                            
        # Delete below threshold reactions if required
        if self.filter_by_freq:
            
            for k in list(self.embeddings.keys()):
                if self.freq_dict[k] < self.filter_threshold:
                    del self.embeddings[k]
                    
        # Normalize input vectors if required
        if self.normalize_vectors:
            for key, value in self.embeddings.items():
                self.embeddings[key] = mu.normalize_vec(value)
        
        
    def calculate_single(self):
        print(str(datetime.now()) + " - Initiating single query")
        
        if self.single_query not in self.embeddings:
            sys.exit('[Error] - no matching embedding for single query: ' 
                     + self.single_query + '. Leave it blank to disable execution.')
            
        single_sims = self.get_similarities_single_input()
        sims = {}
        
        # Sort results from most similar (closest) reactions to least (farthest)
        for metric in self.metrics_list:
            # Eliminate the "self" metric
            single_sims[metric][self.single_query] = -np.inf
            sorted_dict = sorted(single_sims[metric].items(), key = lambda item: -item[1])[:25]
            print('\n' + 'Top ranking ' + metric + ' measures with ' + self.single_query + '\n')
            for i in range(len(sorted_dict)):
                print(sorted_dict[i][0] + ': ' + str(sorted_dict[i][1]))

            sims[metric] = sorted_dict
            
        print(str(datetime.now()) + " - Finishing single query")
        io.write_top_ranking(self.out_dir, self.single_query, sims, '|')
        
        
    def all_query(self):
    
        self.data_dict_all_dists = {}
        self.props_dict_all_dists = {}   
    
        for metric in self.metrics_list:
            
            # Check if a path to precomputed matrices is given, begin all-on-all query if not
            if (self.complete_distance_matrices_path == ''):
                print(str(datetime.now()) + ' - Starting ' + metric + ' for all-on-all query') 
                full_distmat = self.para_process_distmats_for_all(metric)
                
                if self.write_out_all:
                    # Save calculated matrices to CSVs if needed
                    print(str(datetime.now()) + " - Writing complete distance matrix of " + metric)
                    io.write_similarity_matrix(self.out_dir, metric + '_complete_distmat', full_distmat, '|')
                
                # Slightly alter the matrices into the same form as if they were read from precomputed CSVs
                print(str(datetime.now()) + " - Processing complete distance matrix of " + metric)
                full_distmat =  dict(zip(list(full_distmat.keys()), [np.array(list(x.values())) for x in list(full_distmat.values())]))
                
            else:
                # If available, read the precomputed CSVs
                # Has no particular effect when calculate_sims_for_valid = False
                print(str(datetime.now()) + " - Processing complete distance matrix of " + metric)
                full_distmat = io.read_vector_dict(self.complete_distance_matrices_path 
                                            + metric + '_complete_distmat.csv', '|', True)
            
            # Further preprocess matrices for parallelized calculations
            # Has no particular effect when calculate_sims_for_valid = False
            np_matrix, distmat_prop, self.reaction_index_dict = pp.preprocess_distmat(full_distmat)
            self.data_dict_all_dists[metric] = np_matrix
            self.props_dict_all_dists[metric] = distmat_prop
            
            
    def valid_query(self):
        
        print(str(datetime.now()) + " - Preprocessing data for validated data query")
        
        self.group_distmats = pp.get_group_distmats_for_valid(
            self.data_dict_all_dists, self.reaction_index_dict, self.validated_dict, 
            self.metrics_list, self.rand_sample_from_groupping)
        
        self.valid_mean_sim = self.para_process_distmats_for_valid()
        
        print(str(datetime.now()) + " - Postprocessing validated data results")
        self.postprocessing()


    def para_process_distmats_for_valid(self):
    
        n_cpu_worker = mp.cpu_count()    
        if (n_cpu_worker >= len(self.group_distmats)):
            n_workers = len(self.group_distmats)
        else:
            n_workers = n_cpu_worker

        final = {}
        main_data = []

        # Initiate a multiprocessing pool of size n_cpu_worker
        # Each pool process will handle the statistical evaluation of the valid and random
        # reaction groups belonging to one drug at a time
        # The returned main_data is a (number of drugs) long list of two element lists,
        # where the first element is the name of the drug, while the second is a dictionary
        # containing the names and results of the statistical evaluations. This format
        # is only for the easier parallel building of main_data, and so after the parallel
        # block it is changed to a proper dictionary of dictionaries
        # maxstaskperchild = 1 ensures that the pool workers are not reused, but restarted
        # this introduces a time overhead, but reduces memory consumption
        with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

            main_data.extend(pool.starmap(para_handle_distmats_for_valid, zip(
                                            it.repeat(self.stats_list),
                                            it.repeat(self.metrics_list),
                                            it.repeat(self.props_dict_all_dists),
                                            it.repeat(self.stat_perm_num),
                                            self.group_distmats.keys(), 
                                            self.group_distmats.values())))
            pool.close()
            pool.join()
            
        for entry in main_data:

            if entry[1] != {}:

                final[entry[0]] = entry[1]

        return final

        
    def postprocessing(self):
    
        # Write out all the statistical evaluation results into a single file
        io.write_validity_comparison_all(self.metrics_list, self.out_dir, 'all', self.valid_mean_sim, '|')
        
        # Write out all the statistical evaluation results into separate files
        for category in self.valid_mean_sim[next(iter(self.valid_mean_sim))].keys():
            io.write_validity_comparison_single(self.metrics_list, self.out_dir, category.lower(), self.valid_mean_sim, category, '|')
        
        # Determine which results will be aggregated by averaging and which one will be in percentile (of passed p-value threshold) 
        to_average = dict.fromkeys(['VALID', 'RAND', 'VALID_NORM', 'RAND_NORM', 'GAIN_NORM', 'GAIN_NORM_POP'])
        to_percent = dict.fromkeys(['VALID_NORMALITY', 'RAND_NORMALITY',
                                    'VALID_SHAPIRO', 'RAND_SHAPIRO',
                                    'VALID_KOLSIR', 'RAND_KOLSIR',
                                    'MEANMWHITNEY_PVAL', 'VARBROWN_PVAL', 
                                    'PERMDISPPAIR_PVAL', 'PERMANOVA_PVAL'])
        
        summed = {}
        sum_helper = {}
        
        # Create a helper dictionary for the easier aggregation of results
        for i in range(len(self.metrics_list)):
            if self.metrics_list[i] not in sum_helper:
                sum_helper[self.metrics_list[i]] = {}
    
            for j, (drug, entry) in enumerate(self.valid_mean_sim.items()):
                for score_type, score in entry.items():
                    if score_type not in sum_helper[self.metrics_list[i]]:
                        sum_helper[self.metrics_list[i]][score_type] = np.zeros((len(self.valid_mean_sim)))
                    
                    sum_helper[self.metrics_list[i]][score_type][j] = score[i]
        
        # Aggregate the results
        for metric, metric_scores in sum_helper.items():
            summed[metric] = {}
            summed[metric]['MIN'] = self.props_dict_all_dists[metric]['min']
            summed[metric]['MAX'] = self.props_dict_all_dists[metric]['max']
            summed[metric]['MEAN'] = self.props_dict_all_dists[metric]['mean']
            summed[metric]['NORMED_MEAN'] = self.props_dict_all_dists[metric]['normed_mean']
    
            for key, value in metric_scores.items():
                if key in to_average:
                    
                    #Aggregating by averaging
                    summed[metric][key] = np.nanmean(value)
    
                elif key in to_percent:
                    pval = value[~np.isnan(value)]
                    if len(pval) > 0:
                        
                        # p-value correction because of multiple tests, by the Benjamini/Hochberg method
                        corr = multipletests(pvals = pval, alpha = 0.05, method = "fdr_bh")
                        # Aggregating by percentile of adjusted p-values being smaller than 0.05
                        summed[metric][key] =  np.count_nonzero(corr[1] <= 0.05) / corr[1].size
                    else:
                        summed[metric][key] = 'nan'
    
        io.write_validity_postprocess(self.out_dir, 'postprocess', summed, '|')
        

#%% Calculate similarity scores for single input
    def get_similarities_single_input(self):
    
        metric_dict = {}
        
        # Loop through the metric list determined by the input paramaters
        # Initiate a multiprocessing pool of size n_cpu_worker
        # Each pool process will handle the calculation of the given metric with the
        # reaction specified in the parameters against one reaction from the data at a time
        # The final metric_dict returned is a dictionary of dictionaries, containing
        # for each metric all the reaction names and the corresponding values.
        for metric in self.metrics_list:
    
            print(str(datetime.now()) + " - Starting single query metric: " + metric)
            
            n_cpu_worker = mp.cpu_count()
            if (n_cpu_worker >= len(self.embeddings)):
                n_workers = len(self.embeddings)
            else:
                n_workers = n_cpu_worker

            main_data = []
            data_indices = list(self.embeddings.values())
            data_keys = list(self.embeddings.keys())


            if metric == 'cosine':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_cosine_single, zip(
                                                    it.repeat(self.embeddings),
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()


            elif metric == 'euclidean':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_euclidean_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))                  
                    pool.close()
                    pool.join()


            elif metric == 'dot':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_dot_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))                    
                    pool.close()
                    pool.join()


            elif metric == 'manhattan':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_manhattan_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))                    
                    pool.close()
                    pool.join()


            elif metric == 'braycurtis':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_braycurtis_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))                    
                    pool.close()
                    pool.join()
                    

            elif metric == 'canberra':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_canberra_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))                    
                    pool.close()
                    pool.join()


            elif metric == 'chebyshev':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_chebyshev_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()
                    

            elif metric == 'correlation':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_correlation_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()
                    

            elif metric == 'minkowski':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_minkowski_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()
                    

            elif metric == 'seuclidean':

                data_var = np.var(data_indices, axis = 0)

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_seuclidean_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    it.repeat(data_var), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()
                    

            elif metric == 'sqeuclidean':

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_sqeuclidean_single, zip(
                                                    it.repeat(self.embeddings),
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()
                    

            elif metric == 'mahalanobis':

                data_mean = np.mean(data_indices, axis = 0)
                inv_cov = np.linalg.inv(np.cov(np.transpose(data_indices)))

                with mp.Pool(processes = n_workers) as pool:

                    main_data.extend(pool.starmap(sm.calculate_mahalanobis_single, zip(
                                                    it.repeat(self.embeddings), 
                                                    it.repeat(self.single_query), 
                                                    it.repeat(data_indices), 
                                                    it.repeat(data_mean),
                                                    it.repeat(inv_cov), 
                                                    range(len(self.embeddings)))))
                    pool.close()
                    pool.join()
                    
                    
            elif metric == 'meanminpath':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_meanminpath_single, zip(
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys), 
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'primaryminpath':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_primaryminpath_single, zip(
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys), 
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'meanpath':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_meanpath_single, zip(
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys), 
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'primarypath':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_primarypath_single, zip(
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys), 
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'smqmeanminpath':

                if self.single_query in self.smq_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_smqmeanminpath_single, zip(
                                                        it.repeat(self.smq_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys), 
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'resnikmeddra':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_resnikmeddra_single, zip(
                                                        it.repeat(self.ic_meddra),
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys),
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'resniksmq':

                if self.single_query in self.smq_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_resniksmq_single, zip(
                                                        it.repeat(self.ic_smq),
                                                        it.repeat(self.smq_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys),
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'jiangconrathmeddra':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_jiangconrathmeddra_single, zip(
                                                        it.repeat(self.ic_meddra), 
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys),
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'jiangconrathsmq':

                if self.single_query in self.smq_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_jiangconrathsmq_single, zip(
                                                        it.repeat(self.ic_smq), 
                                                        it.repeat(self.smq_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys),
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'linmeddra':

                if self.single_query in self.meddra_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_linmeddra_single, zip(
                                                        it.repeat(self.ic_meddra), 
                                                        it.repeat(self.meddra_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys),
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            elif metric == 'linsmq':

                if self.single_query in self.smq_sep_hier_dict:

                    with mp.Pool(processes = n_workers) as pool:

                        main_data.extend(pool.starmap(sm.calculate_linsmq_single, zip(
                                                        it.repeat(self.ic_smq), 
                                                        it.repeat(self.smq_sep_hier_dict), 
                                                        it.repeat(self.single_query), 
                                                        it.repeat(data_keys),
                                                        range(len(self.embeddings)))))
                        pool.close()
                        pool.join()
                else:
                    main_data = np.array(np.ones(len(self.embeddings)) * -np.inf)
                    

            zip_iterator = zip(data_keys, main_data)
            metric_dict[metric] = dict(zip_iterator)

        return metric_dict
    
    
#%% Calculate similarity scores for all inputs
    def para_process_distmats_for_all(self, metric):
        
        n_cpu_worker = mp.cpu_count()
        if (n_cpu_worker >= len(self.embeddings)):
            n_workers = len(self.embeddings)
        else:
            n_workers = n_cpu_worker

        data_keys = list(self.embeddings.keys())
        data_indices = list(self.embeddings.values())
        main_data = []

        if metric == 'cosine':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_cosine_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'euclidean':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_euclidean_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'dot':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_dot_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'manhattan':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_manhattan_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'braycurtis':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_braycurtis_for_all, zip(
                    it.repeat(self.embeddings), 
                    it.repeat(data_indices), 
                    range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'canberra':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_canberra_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'chebyshev':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_chebyshev_for_all, zip(
                        it.repeat(self.embeddings), 
                        it.repeat(data_indices), 
                        range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'correlation':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_correlation_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'minkowski':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_minkowski_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'seuclidean':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                data_var = np.var(data_indices, axis = 0)

                main_data.extend(pool.starmap(sm.calculate_seuclidean_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                it.repeat(data_var), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'sqeuclidean':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_sqeuclidean_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'mahalanobis':

            data_mean = np.mean(data_indices, axis = 0)
            inv_cov = np.linalg.inv(np.cov(np.transpose(data_indices)))

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_mahalanobis_for_all, zip(
                                                it.repeat(self.embeddings), 
                                                it.repeat(data_indices), 
                                                it.repeat(data_mean), 
                                                it.repeat(inv_cov), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'meanminpath':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_meanminpath_for_all, zip(
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'primaryminpath':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_primaryminpath_for_all, zip(
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()


        elif metric == 'meanpath':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_meanpath_for_all, zip(
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'primarypath':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_primarypath_for_all, zip(
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'smqmeanminpath':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_smqmeanminpath_for_all, zip(
                                                it.repeat(self.smq_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()

        elif metric == 'resnikmeddra':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_resnikmeddra_for_all, zip(
                                                it.repeat(self.ic_meddra), 
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'resniksmq':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_resniksmq_for_all, zip(
                                                it.repeat(self.ic_smq), 
                                                it.repeat(self.smq_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                

        elif metric == 'jiangconrathmeddra':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_jiangconrathmeddra_for_all, zip(
                                                it.repeat(self.ic_meddra), 
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                
                
        elif metric == 'jiangconrathsmq':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_jiangconrathsmq_for_all, zip(
                                                it.repeat(self.ic_smq), 
                                                it.repeat(self.smq_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                
                
        elif metric == 'linmeddra':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_linmeddra_for_all, zip(
                                                it.repeat(self.ic_meddra), 
                                                it.repeat(self.meddra_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                
                
        elif metric == 'linsmq':

            with mp.Pool(processes = n_workers, maxtasksperchild = 1) as pool:

                main_data.extend(pool.starmap(sm.calculate_linsmq_for_all, zip(
                                                it.repeat(self.ic_smq), 
                                                it.repeat(self.smq_sep_hier_dict), 
                                                it.repeat(data_keys), 
                                                range(len(self.embeddings)))))
                pool.close()
                pool.join()
                    
        zip_iterator = zip(data_keys, main_data)
        
        return dict(zip_iterator)
    
    
    def get_out_dir(self):
        return self.out_dir     
    

#%% Define function for multiproseccing-based parallelization which cannot be class method
    # Could be placed into /lib/preprocessing.py but here it is thematically more relevant

def para_handle_distmats_for_valid(stats_list, metrics_list, props_dict_all_dists, perm_num, 
                                   drug_name, drug_entry_dict):
    
    print(str(datetime.now()) + ' - Starting ' + drug_name + ' for validated data query')

    final = {}

    final['VALID'] = []
    final['RAND'] = []
    final['VALID_NORM'] = []
    final['RAND_NORM'] = []
    final['GAIN_NORM'] = []
    final['GAIN_NORM_POP'] = []
    
    for stat in stats_list:         
        if (stat == 'normality' or
            stat == 'shapiro' or
            stat == 'kolsir'):

            final['VALID_' + stat.upper()] = []
            final['RAND_' + stat.upper()] = []
        else:
            # final[stat.upper() + '_FSCORE'] = []
            final[stat.upper() + '_PVAL'] = []
            
    if drug_entry_dict == {}:

        for key in final.keys():
            final[key].append('nan')
    else:
    
        for metric in metrics_list:
            np_matrix_combined = drug_entry_dict[metric]
            if (np.isinf(mu.upper_triangular_to_flat(np_matrix_combined)).all()):
                
                for key in final.keys():
                    final[key].append('nan')
                    
                continue
            
            minv = props_dict_all_dists[metric]['min']
            maxv = props_dict_all_dists[metric]['max']
            normed_meanv = props_dict_all_dists[metric]['normed_mean']            
            group_indices = drug_entry_dict['group_indices']
            
            np_matrix_valid = np.take(np_matrix_combined, group_indices[0], 0)
            np_matrix_valid = np.take(np_matrix_valid, group_indices[0], 1)
            np_matrix_rand = np.take(np_matrix_combined, group_indices[1], 0)
            np_matrix_rand = np.take(np_matrix_rand, group_indices[1], 1)
            
            np_vec_valid = mu.upper_triangular_to_flat(np_matrix_valid)
            np_valid = np_vec_valid[np_vec_valid > -np.inf]
            np_vec_rand = mu.upper_triangular_to_flat(np_matrix_rand)
            np_rand = np_vec_rand[np_vec_rand > -np.inf]   
    
            valid_size = np_valid.size
            rand_size = np_rand.size
    
            if (valid_size == 0 or rand_size == 0):
                
                for key in final.keys():
                    final[key].append('nan')
            else:
    
                metric_mean_valid = np_valid.mean()
                metric_mean_rand = np_rand.mean()
                final['VALID'].append(metric_mean_valid)
                final['RAND'].append(metric_mean_rand)
                metric_std_valid = np_valid.std()
                metric_std_rand = np_rand.std()
    
                if (minv != maxv):
                    z_norm_valid = ((metric_mean_valid - minv)
                                    / (maxv - minv))
                    final['VALID_NORM'].append(z_norm_valid)
                    z_norm_rand = ((metric_mean_rand - minv)
                                    / (maxv - minv))
                    final['RAND_NORM'].append(z_norm_rand)
                    final['GAIN_NORM'].append(z_norm_valid - z_norm_rand)
                    final['GAIN_NORM_POP'].append(z_norm_valid - normed_meanv)
                    
                for stat in stats_list:
                    if stat == 'normality':
                        if np_valid.size > 19:
                            k2, p = stats.normaltest(np_valid)
                            final['VALID_NORMALITY'].append(p)
                        else:
                            final['VALID_NORMALITY'].append('nan')
    
                        if np_rand.size > 19:
                            k2r, pr = stats.normaltest(np_rand)
                            final['RAND_NORMALITY'].append(pr)
                        else:
                            final['RAND_NORMALITY'].append('nan')   
                            
                    elif stat == 'shapiro':
                        if np_valid.size > 2:
                            val_shap, val_shap_p = stats.shapiro(np_valid)
                            final['VALID_SHAPIRO'].append(val_shap_p)
                        else:
                            final['VALID_SHAPIRO'].append('nan')
    
                        if np_rand.size > 2:
                            rand_shap, rand_shap_p = stats.shapiro(np_rand)
                            final['RAND_SHAPIRO'].append(rand_shap_p)
                        else:
                            final['RAND_SHAPIRO'].append('nan') 
                            
                    elif stat == 'kolsir':
                        val_kolsir, val_kolsir_p = stats.ks_1samp(np_valid, stats.norm(loc = metric_mean_valid, scale = metric_std_valid).cdf)
                        rand_kolsir, rand_kolsir_p = stats.ks_1samp(np_rand, stats.norm(loc = metric_mean_rand, scale = metric_std_rand).cdf)
                        final['VALID_KOLSIR'].append(val_kolsir_p)
                        final['RAND_KOLSIR'].append(rand_kolsir_p)
                    
                    elif stat == 'meananova':
                        f_score_anova_mean, p_val_anova_mean = mu.calc_anova_for_means([np_valid, np_rand])
                        # final['MEANANOVA_FSCORE'].append(f_score_anova_mean)
                        final['MEANANOVA_PVAL'].append(p_val_anova_mean)
                        
                    elif stat == 'meanwanova':
                        f_score_wanova_mean, p_val_wanova_mean = mu.calc_welch_anova_for_means([np_valid, np_rand])
                        # final['MEANWANOVA_FSCORE'].append(f_score_wanova_mean)
                        final['MEANWANOVA_PVAL'].append(p_val_wanova_mean)
                    
                    elif stat == 'meanbrown':
                        bf_score_mean, bp_val_mean = mu.calc_brown_for_means([np_valid, np_rand])
                        # final['MEANBROWN_FSCORE'].append(bf_score_mean)
                        final['MEANBROWN_PVAL'].append(bp_val_mean)
                        
                    elif stat == 'meanmwhitney':
                        mw_score_mean, mw_val_mean = mu.calc_mannwhitney_for_means([np_valid, np_rand])
                        # final['MEANMWHITNEY_FSCORE'].append(mw_score_mean)
                        final['MEANMWHITNEY_PVAL'].append(mw_val_mean)
                    
                    elif stat == 'varftest':
                        f_score_var, p_val_var = mu.calc_f_for_vars([np_valid, np_rand])
                        # final['VARFTEST_FSCORE'].append(f_score_var)
                        final['VARFTEST_PVAL'].append(p_val_var)
                        
                    elif stat == 'varbartlett':
                        bart_score_var, bart_val_var = mu.calc_bartlett_for_vars([np_valid, np_rand])
                        # final['VARBARTLETT_FSCORE'].append(bart_score_var)
                        final['VARBARTLETT_PVAL'].append(bart_val_var)
                        
                    elif stat == 'varlevene':                            
                        lf_score_var, lp_val_var = mu.calc_brown_or_levene_for_vars([np_valid, np_rand], 'levene')
                        # final['VARLEVENE_FSCORE'].append(lf_score_var)
                        final['VARLEVENE_PVAL'].append(lp_val_var)
                        
                    elif stat == 'varbrown':
                        bf_score_var, bp_val_var = mu.calc_brown_or_levene_for_vars([np_valid, np_rand], 'brown')
                        # final['VARBROWN_FSCORE'].append(bf_score_var)
                        final['VARBROWN_PVAL'].append(bp_val_var)
                        
                    elif stat == 'permdisppair':
                        new_distmat = pp.readjust_matrix_to_distance(metric, np_matrix_combined)
                        if new_distmat is not None:
                            pdf_score, pdp_val = mu.perform_permdisp_pairwise_dist(new_distmat, group_indices, perm_num)
                        else:
                            pdf_score, pdp_val = 'nan', 'nan'
                        # final['PERMDISPPAIR_FSCORE'].append(pdf_score)
                        final['PERMDISPPAIR_PVAL'].append(pdp_val)                                    

                    elif stat == 'permanova':
                        new_distmat = pp.readjust_matrix_to_distance(metric, np_matrix_combined)
                        if new_distmat is not None:
                            pvf_score_var, pvp_val_var = mu.calc_permanova_for_means(new_distmat, group_indices, perm_num)
                        else:
                            pvf_score_var, pvp_val_var = 'nan', 'nan'
                        # final['PERMANOVA_FSCORE'].append(pvf_score_var)
                        final['PERMANOVA_PVAL'].append(pvp_val_var)
                
    return [drug_name, final]


#%% Initiate main sequence

if __name__ == "__main__":
    
    # Start timer for computation time measurements
    start_time = time.time()
    # Read the provided json file from command-line arguments
    input_settings = io.read_json_input(sys.argv[1])
    # Initialize the main class
    analysis = SimAnalysis(input_settings)
    # Initialization of the class creates the output directory, where the input json file can now be copied
    io.copy_to_out_dir(analysis.get_out_dir(), sys.argv[1], 'input.json')
    # Redirect print() and errors so that it will also write into a log.txt at out_dir
    sys.stdout = io.Logger(sys.stdout, analysis.get_out_dir())
    sys.stderr = sys.stdout
    # Begin queries, which includes data reading and preprocessing
    analysis.query()
    # Conclude
    print(str(datetime.now()) + ' - All done!')
    elapsed_time = time.time() - start_time
    print(str(datetime.now()) + " - Only took %02d seconds!" % (elapsed_time))
