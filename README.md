# Term-Deposit-Subscription-Project-ML-

Problem Statement
Banks run phone campaigns to get customers to subscribe to term deposits. Calling everyone is costly and inefficient. This project builds a model to predict which customers are likely to subscribe, so the bank can target the right people.

Data Source

UCI Machine Learning Repository — Bank Marketing Dataset (http://archive.ics.uci.edu/dataset/222/bank+marketing)

41,188 records, 21 features covering customer demographics, campaign details, and macroeconomic indicators. Target variable: y (subscribed: yes/no).


EDA

No missing values; 12 duplicates removed

Heavily imbalanced — 89% no, 11% yes

Students and retirees had the highest subscription rates

March and December campaigns performed best

Macroeconomic features (euribor3m, emp.var.rate) strongly correlated with the target

Feature Engineering
Step      Reason 
Train/test split first:    Prevent data leakage

StandardScaler (numeric):    Normalise feature ranges

OneHotEncoder (categorical):     Convert categories to numeric

SMOTE (training set only):    Fix class imbalance

Modelling

Three classifiers trained with fixed parameters (no hyperparameter tuning):

Decision Tree — max_depth=5

Random Forest — n_estimators=100, max_depth=10

XGBoost — n_estimators=100, learning_rate=0.1, max_depth=5

Metrics Used & Why

Accuracy alone is misleading with imbalanced data (an all-no model scores 89%). More informative metrics:

Metric    Why

Precision:   Avoids wasting calls on unlikely subscribers

Recall:     Captures as many real subscribers as possible

F1-Score:   Balances precision and recall

ROC AUC:   Overall discriminative power across thresholds

Results
Model - Accuracy  -   Precision -  Recall  -  F1  -   ROC AUC

XGBoost  -- 0.910 -- 0.577   -- 0.740 -- 0.648 -- 0.948

Random Forest   --0.882  -- 0.487  --  0.881  -- 0.627  --0.942

Decision Tree  -- 0.862  --  0.443  --  0.881  --  0.590  ---  0.927

XGBoost selected as best model (highest ROC AUC and F1-Score).

Key Findings

The model correctly identifies 74% of actual subscribers, allowing the bank to run targeted campaigns

Macroeconomic conditions (interest rates, employment rate) are the strongest predictors

Students, retirees, and March/December contacts convert at the highest rates

Call duration is highly predictive but only known after the call, limiting real-world use


What I Learned

Accuracy misleads on imbalanced data — ROC AUC and F1 are more honest metrics

SMOTE must only be applied to training data — applying it before splitting contaminates the test set

Preprocessing must be fit on training data only — fitting on the full dataset leaks test information

OHE breaks column names — ohe.get_feature_names_out() is needed for correct feature importance 
labels

Simple models are often good enough — fixed parameters ran much faster than GridSearchCV with comparable results
