# Design Decisions for Task 1

1. Encode the categorical variables using MAP as sklearn cannot handle categorical variables.
2. Impute the missing categorical and continuous variables using sklearn Imputer as sklearn cannot handle missing values.
3. Use sklearn train_test_split to divide the data into train, validation and test to ensure to there is overfitting and to avoid optimization bias.
4. Try different classification models & identify the best performing model.
5. Perform hyperparameter optimzation to tune the best performing model. The best model was selected for task1 is GradientBoostingClassifier.
6. Perform feature selection using the optimized model and sklearn pipeline.

