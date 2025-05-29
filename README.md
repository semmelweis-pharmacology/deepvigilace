# Deepvigilace project

Deepvigilace is a software framework for creating, analysing and using adverse event vector representations, a.k.a. embeddings, for pharmacovigilance purposes. See our paper **()** for an in-depth description of the project.

## Preliminaries

The creation of the representation vectors requires pre-processed adverse event reports, while their use for causality assessment needs additional labeled drug–ADR (adverse drug reaction) data and drug features. These files are available at **()**, along with other supplementary data and visualizations. In this repository, we provide the code necessary for replicating the results of our paper from these files.

## Usage

Each of the following scripts correspond to a specific module of the project. These scripts can be run with a single argument, which is the full path to their associated JSON input, containing all the necessary input file paths and parameters. See our paper, the associated JSON templates, and the documentations in these scripts themselves for further details. All scripts were executed on an Ubuntu (22.04.2 LTS) system. See the provided ```env.yml``` file for the Conda virtual environment and exact packages used during the project.

```
python3 -u {{script.py}} {{input.json}}
```

| Module | Script | Description* |
| ------ | ------ | ------ |
| Embedding | ```embedding.py``` | Responsible for creating the vector representation of adverse events via noise contrastive estimation from pre-processed adverse event reporting data. Input files:<br /><ul><li> A context-sampled report file (e.g. ```./report_corpus/context_sampled_reports_reaction2all.csv```)</li></li><li>The corresponding target and context frequency files (e.g. ```./frequencies/target_frequency_reaction2all.csv``` and ```./frequencies/context_frequency_reaction2all.csv```)</ul> |
| Dimensionality Reduction and Clustering | ```clustering.py``` | Performs dimensionality reduction with t-SNE, then clustering via HDBSCAN. Input files:<br /><ul><li>The selected embedding (e.g. ```./embeddings/nsg_reaction2all_embedding.csv```)</li></ul> |
| Dispersion and Similarity Analyses | ```sim_analysis.py``` | Calculates similarities/distances among the embedding vectors from which it can output the top neighbours of a selected adverse event or analyse the entire embedding space thorugh various statistical tests, as well as our "gain norm" measure. Input files:<br /><ul><li> The selected embedding (e.g. ```./embeddings/nsg_reaction2all_embedding.csv```)</li></li><li>A dictionary derived from MedDRA (optional, not provided)</li><li>A dictionary derived from MedDRA-SMQ (optional, not provided)</li><li>The corresponding (target) frequency file (e.g. ```./frequencies/target_frequency_reaction2all.csv```) </li><li>An IME reaction list (```./other/meddra_ime_list_from_ema.csv```)</li><li>Files containing validated drug–ADR pairs (e.g. ```./validated_data/sider.csv```)</ul> |
| Classifier | ```classifier.py``` | Trains and tests a binary classifier to predict the causality between drugs and adverse events. Input files:<br /><ul><li> A report corpus file containing unique drug–event pairs (```./report_corpus/reports_for_classifier.csv```)</li></li><li>The selected embedding, as adverse event features (e.g. ```./embeddings/nsg_reaction2all_embedding.csv```)</li></li><li>List of "chemical" properties of drugs (```./drug_features/drug_features_chem.csv```)</li></li><li>List of "biological" properties of drugs (```./drug_features/drug_features_bio.csv```)</li></li><li>Files containing benchmark drug–ADR pairs, both positively and negatively associated (e.g. ```./benchmark_data/omop.csv```)</ul> |
| Classifier in Inference mode | ```classifier_tester.py``` | Tests a trained binary classifier on the provided test files if labeled, or runs the classifier in inference mode without performance evaluation if not. Input files:<br /><ul><li> The selected embedding, as adverse event features (e.g. ```./embeddings/nsg_reaction2all_embedding.csv```)</li></li><li>List of "chemical" properties of drugs (```./drug_features/drug_features_chem.csv```)</li></li><li>List of "biological" properties of drugs (```./drug_features/drug_features_bio.csv```)</li></li><li>Files containing test drug–ADR pairs, either labeled or not (e.g. ```./benchmark_data/omop.csv```)</li></li><li>A pretrained model (```./classifier_results/best_model/model.h5```)</li></li><li>The corresponding standard scaler (```./classifier_results/best_model/std_scaler.bin```)</ul> |

_*All file paths denoted here are relative to the path to the directory of the supplementary data._


## Citation

```
@article {,
	author = {},
	title = {},
	elocation-id = {},
	year = {},
	doi = {},
	publisher = {},
	URL = {},
	eprint = {},
	journal = {}
}
```
