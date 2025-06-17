# Bimodal Peptide Collision Cross Section Distributions Reflect Two Stable Conformations in the Gas Phase

Repsitory containing the trained models and the code for the paper mentioned on the title.

## Data Availability

Download the concatenated MaxQuant output file `evidence-03122024.csv` from [here](https://data.mendeley.com/datasets/szrn5srhyw) and place it under anew folder `/data`. In the same repository you can find the aligned train and test sets, i.e., the files `evidence_aligned_trainv2660.csv` and `evidence_test_fromRawFiles_2pop_v2660.csv`. Place them as well under `/data` if you dont want to do the alignment following `K0_alignment.ipynb`.

## Merging of multiple runs
Because we are integrating results from different MaxQuant runs we need to allign them. This is done in the notebook `K0_alignment.ipynb`.

## Paper Figures

* The figures of the section `Bi-modality of peptide collision cross section distribution` are in the notebook `section_1.ipynb`
* The development and plots for the geometric scattering fit are in the notebook `fit_geometric_scaterring.ipynb`
* The development and plots for the geometric fit are in the notebook `fit_geometric.ipynb`
* The development and plots for the empirical fit are in the notebook `fit_empirical.ipynb`
* The Machine Learning Regressor is trained in the notebook `training-prediction.ipynb`
* The MaxDIA analysis is in the notebook `maxdia.ipynb`

