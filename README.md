# Bimodal Peptide Collision Cross Section Distributions Reflect Two Stable Conformations in the Gas Phase

Repsitory containing the trained models and the code for the paper mentioned on the title.

## Data Availability

Download the concatenated MaxQuant output file `evidence-03122024.csv` from [here](https://data.mendeley.com/preview/szrn5srhyw?a=3381e6af-4c79-4f92-be3a-9ca13ef0c9fa) and place it under anew folder `/data`

## Merging of multiple runs
Because we are integrating results from different MaxQuant runs we need to allign them. This is done in the notebook `K0_alignment.ipynb`

## Paper Figures

* The figures of the section `Bi-modality of peptide collision cross section distribution` are in the notebook `section_1.ipynb`
* The development and plots for the geometric scattering fit are in the notebook `fit_geometric_scaterring.ipynb`
* The development and plots for the geometric fit are in the notebook `fit_geometric.ipynb`
* The development and plots for the empirical fit are in the notebook `fit_empirical.ipynb`
* The Machine Learning Regressor is trained in the notebook `training-prediction.ipynb`
* The MaxDIA analysis is in the notebook `madia.ipynb`

