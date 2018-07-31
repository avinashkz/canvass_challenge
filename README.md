# Design Decisions for Task 1

1. Processed the target variable to predict failure between 0 and 40 days.

2. Preprocessed the data (features)
    - Encode the categorical variables using `map` as sklearn cannot handle categorical variables. (I have assumed that the differences between all the levels are the same.)
    - Impute the missing values using other features as there was a significant correlation between features. 
    - Using `RandomForestRegressors` for continuous /discrete features.
    - Using `RandomForestClassifier` for categorical features.
    - Impute missing values using Mean(continuous /discrete)/Mode(categorical) if top correlated features are all missing.
3. Use sklearn `train_test_split` to divide the data into train and test to ensure that the test data is isolated from the training data and to avoid optimization bias.
4. Upsample the training data using the `imbalance-learn` package. Classification models returned around 83% recall without upsampling the data.
5. Divide the upsampled data into train and validation using train test split to avoid overfitting.
6. Try different classification models & identify the best performing model using the validation set.
7. Perform hyperparameter optimization to tune the best performing model using `RandomizeSearchCV`. The best model selected for task1 is Random Forest Classifier.
8. Perform feature selection using the optimized model and sklearn pipeline.
9. Since I was getting good accuracy and recall score, I did not spend time on neural networks for this task.

The code for task 1 can be found in [task1.ipynb](src/Task1.ipynb)

The code to process data for setting up the target variable can be found in [preprocessing1.Rmd](src/preprocessing1.Rmd)

**Sample usage [Task1-Sample.ipynb](src/Task1-Sample.ipynb)**

# Design Decisions for Task 2

1. Processed the target variable to predict 6 hours ahead.

2. Preprocessed the data (features)
    - Encode the categorical variables using `map` as sklearn cannot handle categorical variables. (I have assumed that the differences between all the levels are the same.)
    - Impute the missing values using other features as there was a significant correlation between features. 
    - Using `RandomForestRegressors` for continuous /discrete features.
    - Using `RandomForestClassifier` for categorical features.
    - Impute missing values using Mean(continuous /discrete)/Mode(categorical) if top correlated features are all missing.
    - Split all the categorical features into binary variables.
3. Develop and compare various sklearn regression models and identify the best model using the validation set.  The best model selected for task2 is `RandomForestRegressor`.
4. Perform hyperparameter optimization using `GridSearchCV`
5. Perform feature selection using the sklearn `pipeline` and the best performing non-deep learning model.
6. Attempt to improve the performance of Random Forest model using Neural Networks.
7. Tune hyperparameters for Keras models using `GridSearchCV` and `RandomizedSearchCV`.
8. Save the model in pickle format.

The code for task 2 can be found in [task2.ipynb](src/Task2.ipynb)

The code for NN using Keras -> [Task2_NN.ipynb](src/Task2_NN.ipynb)

The code to process data for setting up the target variable can be found in [preprocessing2.Rmd](src/preprocessing2.Rmd)

**Sample usage [Task2-Sample.ipynb](src/Task2-Sample.ipynb)**
