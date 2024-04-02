# Electric Vehicle Purchase Prediction Data Preprocessing Model Training and Prediction


## 1. Importing Libraries:
The code starts by importing necessary libraries such as Pandas for data manipulation, NumPy for numerical operations, Matplotlib and Seaborn for data visualization, and various modules from Scikit-learn for machine learning tasks.

## 2. Loading Data:
Data loading is performed using Pandas' read_csv function for both the training dataset (EV_train.csv) and the test dataset (EV_X_test.csv).

## 3. Data Exploration:
Basic exploratory data analysis (EDA) tasks are executed, including checking the structure of the datasets (info()), inspecting the first few rows (head()), identifying duplicate rows, and detecting missing values.

## 4. Categorical Feature Analysis:
The code analyzes categorical features by generating frequency and relative frequency tables for each column, excluding specific columns related to geographical information.

## 5. Preliminary Data Preprocessing:
Initial preprocessing steps are applied to specific columns like 'TownToFastChgDriveTime' and 'HwyFastChgDistance', where certain values are replaced and converted to integers.

## 6. Feature Engineering:

This section involves preprocessing categorical features such as 'race' and 'state' by merging low-frequency categories and applying one-hot encoding.

Other features like 'employment', 'housit', 'residence', etc., are also preprocessed using one-hot encoding and value recoding.

Geographical encoding is performed on the 'zipcode' feature by extracting the first three digits and analyzing their distribution.

Numerical features are analyzed for skewness, outliers, and correlations.

## 7. Feature Selection:
Feature selection techniques such as Recursive Feature Elimination (RFE) and independence tests are employed to select relevant features for modeling.

## 8. Classification Models:

Two classification models, Random Forest Classifier and XGBoost Classifier, are trained and evaluated.

Random Forest Classifier is optimized using RandomizedSearchCV and GridSearchCV to find the best hyperparameters.

XGBoost Classifier is trained with predefined parameters.

Model performance is evaluated using accuracy, confusion matrix, classification report, and ROC-AUC score.

# 9. Test Set Prediction:

The best-performing model (XGBoost Classifier) is utilized to make predictions on the test set (EV_X_test.csv).
Predictions are saved to a DataFrame and exported as a CSV file following specific naming conventions.
