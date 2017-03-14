# Udacity Machine Learning Nanodegree
## Capstone project

This project uses the MIMIC-III database, accessible via:
https://mimic.physionet.org/

To properly run the project from the included files, a local postgres SQL server must be installed and the MIMIC-III database must be set up as describe. in https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic.

An SQL materialized view was extracted from the database as defined in all_data.sql.

python libraries used:

numpy - numerical operation

pandas - dataframe handling

os - general operating system operations

psycopg2 - Used to access a locally installed postgresql server and 
perform sql queries. 

xgboost - eXtreme Gradient Boosted trees. Classifier implementation.

scikit-learn - Used for hyperparameter optimization and performance 
metrics evaluation.

scipy - interp function used during plotting of the ROC curve.

matplotlib - visualization

seaborn - visualization