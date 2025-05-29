#%% Import libraries

import sys
import time

import pandas as pd

from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN

# Project specific modules
from lib import io_file_handling as io


#%% Define main class

class Clustering():
    
    """
    The Clustering class encapsulates the dimension reduction of the embedding
    vectors with t-SNE and their clustering with HDBSCAN.

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
        
    tsne_dimensions: int
        This parameter determines the "number of components" for t-SNE, meaning
        the size of the target dimension it will map the input embedding 
        vectors into. Recommended 2 or 3, making the results easily 
        visualizable in either 2D or 3D plots.
    
    tsne_perplexity: int
        This parameter determines the number of neighbors t-SNE considers for
        each embedding vector in the original vector space. Optimization
        consists of placing the embedding vectors into the target space in such
        a way that their original neighbors remain in their local proximity, 
        similarly distributed, as much as possible. Recommended value is 30-50.
    
    tsne_iterations: int
        This parameter determines the number of maximum training iterations or 
        epochs t-SNE shall do (the parameter "n_iter_without_progress" may
        terminate it before reaching this). Recommended value is 1000-3000.
    
    tsne_metric: str
        This parameter determines the distance metric used by t-SNE for
        selecting and measuring the distance of the closest neighbors for each
        vector in the original space. As the reaction embedding vectors
        originate from neural networks with dot and cosine based outputs, it is
        highly recommended to use cosine here as well. See the original scikit 
        learn documentation for the other options.
        
    tsne_random_state: int
        This parameter influences the randomness of t-SNE, enabling 
        reproducible results.
    
    hdbscan_minimum_cluster_size: int
        This parameter determines a threshold for the number of points under 
        which HDBSCAN does not consider them to be a meaningful cluster during
        the construction of the condensed cluster tree. Strongly influences
        the sizes and number of resulting clusters. Recommended value is 5-15.
    
    hdbscan_minimum_samples: int
        This parameter determines the size of the area used for density
        estimation, and thus the robustness of it, during the transformation
        of the distances into mutual reachability distances. Influences how
        strict the separation of clusters from noise is, with lower values
        resulting in less noise and more "fuzzy" clusters. Recommended value
        is "None" which is the default and makes it equal to the minimum
        cluster size.
    
    hdbscan_metric: str
        This parameter determines the distance metric used by HDBSCAN during 
        the first steps of calculating the mutual reachability distances and 
        creating the minimum spanning tree. With the reaction embeddings
        transformed with t-SNE, it is recommended to use Euclidean distance.
        See the original scikit learn documentation for the other options.
        
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
        word vectors  (numpy arrays, float64) as values. This serves as the
        main input data for the clustering.
        
    tsne_results: sklearn.manifold.TSNE() object
        Contains the fitted t-SNE object (i.e. coordinates) which is directly 
        passed down to the HDBSCAN model.


    Methods
    ----------
    read_settings():
        Extracts the input parameters from the provided json object into class
        variables. Furthermore, creates the output folder at the designated
        destination. Executed during class initialization. See out_dir and the 
        rest of the parameter descriptions for further details.
        
    train():
        The main method of the Clustering class, intended to start the entire
        process, first with t-SNE then with HDBSCAN.
        
    read_files():
        Reads the input data file into memory, see embedding_path for further 
        details.
        
    train_tsne():
        Initiates the t-SNE model with the provided parameters and trains it
        (fits) on the input embedding vectors, which results in the transformed
        coordinates. Combined with the reaction MedDRA PTs, the coordinates
        are written into a CSV file in the output folder for reuse and
        visualization purposes.
        
    train_hdbscan():
        Initiates the HDBSCAN model with the provided parameters and trains it
        on the transformed coordinates, made by t-SNE. The clustering results
        are written into a CSV file in the output folder for visualization
        purposes.
        
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
        self.embedding_path = self.input_settings["oSourceList"]["oEmbedding"]
        
        # t-SNE-specific parameters
        self.tsne_dimensions = self.input_settings["oParameterList"]["oTSNE"]["oDimensions"]
        self.tsne_perplexity = self.input_settings["oParameterList"]["oTSNE"]["oPerplexity"]
        self.tsne_iterations = self.input_settings["oParameterList"]["oTSNE"]["oIterations"]
        self.tsne_metric = self.input_settings["oParameterList"]["oTSNE"]["oMetric"]
        self.tsne_random_state = self.input_settings["oParameterList"]["oTSNE"]["oRandomState"]
        
        # HDBSCAN-specific parameters
        self.hdbscan_minimum_cluster_size = self.input_settings["oParameterList"]["oHDBSCAN"]["oMinClusterSize"]
        self.hdbscan_minimum_samples = self.input_settings["oParameterList"]["oHDBSCAN"]["oMinSamples"]
        self.hdbscan_metric = self.input_settings["oParameterList"]["oHDBSCAN"]["oMetric"]
        
        # Output path
        self.output_path = self.input_settings["oOutputPath"]
        self.project_name = self.input_settings["oProjectName"].lower()
        
        self.out_dir = io.make_out_dir(self.output_path, self.project_name)

    
    def train(self):
        
        self.read_files()
        self.train_tsne()
        self.train_hdbscan()
        
        
    def read_files(self):
        print(str(datetime.now()) + ' - Reading input files')
        
        self.embedding = pd.read_csv(self.embedding_path, sep = "|", index_col = 0, header = None)
        
        
    def train_tsne(self):
        print(str(datetime.now()) + ' - Training t-SNE')
        
        tsne = TSNE(
                        n_components = self.tsne_dimensions,
                        perplexity = self.tsne_perplexity,
                        random_state = self.tsne_random_state,
                        n_iter = self.tsne_iterations,
                        metric = self.tsne_metric,
                        verbose = 1
                        )
        try:
            
            self.tsne_results = tsne.fit_transform(self.embedding)
            
        except Exception as e:
            print('[Error] t-SNE was misparametrized. Check scikit-learn documentation for viable parameters.')
            sys.exit(e)

        print(str(datetime.now()) + ' - Writing t-SNE results')
        tsne_dict = dict(zip(self.embedding.index, self.tsne_results))
        io.write_vector_dict(self.out_dir, 'tsne_coords', tsne_dict, '|')
        
        
    def train_hdbscan(self):
        print(str(datetime.now()) + ' - Training HDBSCAN')
        
        hdbscan = HDBSCAN(
                            min_cluster_size = self.hdbscan_minimum_cluster_size,
                            min_samples = self.hdbscan_minimum_samples,
                            metric = self.hdbscan_metric
                            )
        
        try:
            
            hdbscan.fit(self.tsne_results)
            
        except Exception as e:
            print('[Error] HDBSCAN was misparametrized. Check scikit-learn documentation for viable parameters.')
            sys.exit(e)
        
        print(str(datetime.now()) + ' - Writing HDBSCAN results')
        hdbscan_dict = dict(zip(self.embedding.index, hdbscan.labels_))
        io.write_simple_dict(self.out_dir, 'clusters', hdbscan_dict, '|')
        

    def get_out_dir(self):
        return self.out_dir


#%% Initiate main sequence

if __name__ == "__main__":
    
    # Start timer for computation time measurements
    start_time = time.time()
    # Read the provided json file from command-line arguments
    input_settings = io.read_json_input(sys.argv[1])
    # Initialize the main class
    clustering = Clustering(input_settings)
    # Initialization of the class creates the output directory, where the input json file can now be copied
    io.copy_to_out_dir(clustering.get_out_dir(), sys.argv[1], 'input.json')
    # Redirect print() and errors so that it will also write into a log.txt at out_dir
    sys.stdout = io.Logger(sys.stdout, clustering.get_out_dir())
    sys.stderr = sys.stdout
    # Begin model training, which includes data reading and preprocessing
    clustering.train()
    # Conclude
    print(str(datetime.now()) + ' - All done!')
    elapsed_time = time.time() - start_time
    print(str(datetime.now()) + " - Only took %02d seconds!" % (elapsed_time))

