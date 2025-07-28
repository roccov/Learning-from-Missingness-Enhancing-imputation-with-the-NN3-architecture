

# Learning from Missingness: Enhancing imputation with the NN3 architecture

- Conducted data preprocessing and feature engineering using Compustat accounting data, JKP monthly factors returns and CRSP Security-Level Prices database from 1961 to 2024

- Building a three stage pipeline model that combines Random Forest-based feature selection, hyperparameter optimization and a Neural Network trained on rolling windows to     predict monthly US common stock returns

- Designed and tested varying regularization schemes; introduced a NaN Flag Matrix to encode missingness as a predictive signal, enhancing imputation and model performance.

- Achieved a net-of-cost positive return over a 40-year backtest period through a systematic trading strategy driven by the model’s forecasts.



## The repository is structured in the following way:

1. data_processing_and_features_engineering : Data wrangling, feature engineering and merging

2. LASSO : Lasso regressions

3. NN3_HR : Heavily regularized NN3 architecture 

4. NN3_MR : Mildly regularized NN3 architecture

5. NN3_MR_F : Mildly regularized NN3 architecture with Nan Flags adjonction

------------- HOW TO RUN THE NN3 CODE -------------------

To run the code, make sure the data file is inside of the folder. CHange the parameters as you wish in the config.py file.

Then, Simply execute the main.py file in the terminal.


This project was carried out as part of the Machine Learning for Finance course taught by Professor Semyon Malamud at the École Polytechnique Fédérale de Lausanne (EPFL), in collaboration with:

Rocco Pio Lorenzo Ventruto lien des githubs respectifs
talula
noah
pyotr

This repository is a personal, cleaned-up version of the original group project, with additional comments and refinements for portfolio purposes.
