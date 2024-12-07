# congress_districts
PREDICTING 2024 CONGRESSIONAL DISTRICT OUTCOMES USING POLLING DATA AND DIMENSIONALITY REDUCTION ON AMERICAN COMMUNITY SURVEY DATA

Utilizing singular value decomposition (SVD) with an automated selection criterion, which captures 97.5% of the variation in the data, American Community Survey (ACS) 5-Year Estimates that were originally comprised of 7,378 separate features were reduced to a mere 16 components. On fitting a baseline model with hyperparameter tuning but that omits the relevant pre-election polling data, these 16 components yield a training set overall accuracy of 96.7%, a validation set overall accuracy of 84.1%, and a testing set overall accuracy of 87.4%. Given the shifting congressional district lines between the naively chosen training and validation set year (i.e., 2020) and testing set year (i.e., 2022), coupled with the fact that the latter was also a midterm election cycle where the composition of the vote typically differs from the general election cycle in both number and demographically, these early results display just how powerful a technique like SVD can be by itself for distilling from a large amount of data those salient features that are encoded within it.

![metrics](images/metrics.png)

Included in this project are three main component files: **data_prep.py**, **development.py**, and **pipeline.py**

The script **data_prep.py** retrieves the ACS variables using the Census API, prepares the datasets, performs the SVD, and determines the top feature contributions in the calculated components. It requires a registered Census API key (see: https://api.census.gov/data/key_signup.html). 

The script **development.py** is for local development and allows the user to run the XGBoost model and metrics without the need for an AWS account.

The script **pipeline.py** is the main driver of a SageMaker Pipelines machine learning operations (MLOps) environment and requires an AWS account.

Note: This is an ongoing, active project. Taking a cue from other popular polling aggregate outcome predictors, it is intended as a proof of concept to show the development and deployment of an XGBoost classification model using the Census API, Amazon Code Editor (Visual Studio Code-OSS), and Amazon Web Services (AWS) S3 buckets, EC2 instances, and SageMaker Pipelines. Stay tuned.