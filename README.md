

# Learning from Missingness: Enhancing imputation with the NN3 architecture to predict monthly US stock returns


## Highlights: 

- Conducted data preprocessing and feature engineering using Compustat accounting data, JKP monthly factors returns and CRSP Security-Level Prices database from 1961 to 2024

- Building a three stage pipeline model that combines Random Forest-based feature selection, hyperparameter optimization and a Neural Network trained on rolling windows to     predict monthly US common stock returns

- Designed and tested varying regularization schemes; introduced a **NaN Flag Matrix** to encode missingness as a predictive signal, enhancing imputation and model performance.

- Achieved a **net-of-cost positive return** over a 40-year backtest period through a systematic trading strategy driven by the model’s forecasts.



## Repository Structure:

00_Report.pdf : Full academic report

01_Data_Processing_And_Features_Engineering : Data wrangling, feature engineering and merging

02_LASSO : Lasso regressions

03_NN3_HR : Heavily regularized NN3 architecture 

04_NN3_MR : Mildly regularized NN3 architecture

05_NN3_MR_F :  Mildly regularized NN3 architecture with Nan Flags adjonction



Full prediction results are available upon request or can be regenerated using the pipeline.

## Running the NN3 code:

To run the code, make sure the data file is inside of the folder. CHange the parameters as you wish in the config.py file.

Then, Simply execute the main.py file in the terminal.

## Collaborators: 
This project was carried out as part of the Machine Learning for Finance course taught by Professor Semyon Malamud at the Ecole Polytechnique Fédérale de Lausanne (EPFL), in collaboration with:

Rocco Pio Lorenzo Ventruto [https://github.com/roccov]

Tallula Graber [https://github.com/Tallulaa]

Noah Louis Truttmann [https://github.com/NoahTruttmann]

Piotr Kleymenov [https://github.com/PiotrKley259]

This repository is a personal, cleaned-up version of the original group project, with additional comments and refinements for portfolio purposes.
