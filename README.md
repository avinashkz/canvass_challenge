# Design Decisions for Task 1

1. Encode the categorical variables using `map` as sklearn cannot handle categorical variables. (I have assumed that the differences between all the levels are the same.)
2. Impute the missing categorical and continuous variables using sklearn `Imputer` function as sklearn cannot handle missing values.
3. Use sklearn `train_test_split` to divide the data into train and test to ensure that the test data is isolated from the training data and to avoid optimization bias.
4. Upsample the training data using the `imbalance-learn` package. Classification models returned around 30% recall without upsampling the data.
5. Divide the upsampled data into train and validation using train test split to avoid overfitting.
6. Try different classification models & identify the best performing model using the validation set.
7. Perform hyperparameter optimization to tune the best performing model using `RandomizeSearchCV`. The best model selected for task1 is Random Forest Classifier.
8. Perform feature selection using the optimized model and sklearn pipeline.

The code for task 1 can be found in [task1.ipynb](src/task1.ipynb)

The code for task 1 (attempt to predict 40 minutes ahead)can be found in [task1-40days.ipynb](src/task1-40days.ipynb)

The code to process data for predicting 40 minutes ahead can be found in [preprocessing.Rmd](src/preprocessing.Rmd)



# Design Decisions for Task 2

1. I assumed that the value of y given is the traffic 6 hours ahead.
2. Encoded the categorical variables as sklearn cannot handle them. (I have assumed that the differences between all the levels are the same.)
3. Imputed the categorical and continuous variable as sklearn cannot handle missing values.
    - Since there is a significant correlation between some of the features, I attempted to predict the categorical variables `label` as it would be more accurate than replacing with the most frequent value.
    - Also, I tried to remove all the missing values. This gave better results than imputing the missing values.
4. Develop and compare various sklearn regression models and identify the best model using the validation set.  The best model selected for task2 is GradientBoostingRegressor.
5. Perform hyperparameter optimization using GridSearchCV
6. Perform feature selection using the sklearn `pipeline` and the best performing model
7. Save the model in pickle format.

The code for task 2 can be found in [task2.ipynb](src/task2.ipynb)
