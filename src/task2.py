# Importing dependencies
import numpy as np
import pandas as pd

# Data preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

# Classification models
from sklearn.ensemble import RandomForestClassifier

# Regression models for imputation
from sklearn.ensemble import RandomForestRegressor

# To save the final model
import _pickle as pickle

def process_data(X, refit = False):
    
    """
    Args:
        X : Pandas Dataframe
            A dataframe containing all the features for task 2.
        refit : Boolean
            Refit all the models if `True`.
            
    Returns:
        df : Pandas Dataframe
            A dataframe containing all the features with missing values imputed.
    """

    fixed_values = {}
    
    all_data = X.loc[:, ['co_gt', 'nhmc', 'c6h6',
                         's2', 'nox', 's3', 'no2',
                         's4', 's5', 't','rh', 'ah',
                         'level']].copy()
    
    #I have made the assumption that the difference between all the levels are the same. 
    all_data["level"] = all_data["level"].map( {'Very low': -2, 'Low':-1, 'moderate':0, 'High':1, 'Very High':2 } )
    
    feat = [np.abs(item)[item != 1] for item in [all_data.corr()[item] for item in all_data.columns]]

    for c,columns in enumerate(feat):

        col_name = pd.DataFrame(columns).columns[0]

        print("==== Processing {} ====".format(col_name))

        # Identify the top 3 correlated features for the selected feature
        sorted_col = pd.DataFrame(columns).sort_values(col_name)

        top_corr = sorted_col.iloc[-1].name
        second_corr = sorted_col.iloc[-2].name
        third_corr = sorted_col.iloc[-3].name

        features = [top_corr, second_corr, third_corr]
        
        # Extract target variable and correlated features
        y = all_data.loc[:, col_name]
        x1 = all_data.loc[:, top_corr]
        x2 = all_data.loc[:, second_corr]
        x3 = all_data.loc[:, third_corr]

        if refit:        

            # Fit a model to predict the target variable using each of the correlated feature
            print(" \t Fitting")
            for number, item in enumerate([x1, x2, x3]):   
                X_train = item[np.invert(y.isnull() | item.isnull())].values.reshape(-1, 1)
                y_train = y[np.invert(y.isnull() | item.isnull())]

                # Select Classifier or Regression model based on the number of unique values
                if(len(y.unique()) <= 7):
                    rf_model = RandomForestClassifier()
                    
                    # Boosting categorical variables that has values between 0 and 1.
                    if y_train[0] < 1 : 
                        y_train = y_train*100
                    
                    # Classification models cannot handle float.
                    y_train = y_train.astype('int')
                
                    # Set most frequent item as back-up if top 3 correlated features are missing.
                    fixed_values[col_name] = mode(y)

                else:
                    rf_model = RandomForestRegressor()
                    
                    # Setting mean of the variable as back-up if top 3 correlated features are missing.
                    fixed_values[col_name] = np.mean(y)

                # Fitting Random forest model
                rf_model.fit(X_train, y_train)


                #Save the built model
                with open('../results/pickles/{}-{}x.pickle'.format(col_name, number), 'wb') as save_model:
                    pickle.dump(rf_model, save_model)

            # Save the dictionary containing fixed values if all the top 3 correlated features are missing.
            with open('../results/pickles/fixed_values2.pickle', 'wb') as save_model:
                pickle.dump(fixed_values, save_model)

        print(" \t Imputing \n")

        # Data missing selected feature
        
        # Target variable is missing & X1 is present.
        fallback_1 = y.isnull() & np.invert(x1.isnull())
        # Target variable and X1 are missing. But X2 is present
        fallback_2 = y.isnull() & x1.isnull() & np.invert(x2.isnull())
        # Target variable, X1 and X2 are missing. But X3 is present
        fallback_3 = y.isnull() & x1.isnull() &  x2.isnull() & np.invert(x3.isnull())
        # The target variable and all the top 3 correlated features are missing.
        fallback_4 = y.isnull() & x1.isnull() &  x2.isnull() & x3.isnull()
        fallbacks = [fallback_1, fallback_2, fallback_3, fallback_4]

        complete_data = all_data.loc[np.invert(y.isnull())].copy()

        for count, item in enumerate(fallbacks):
            if item.sum().astype("bool") & (count < 3):
                loop_data = all_data.loc[item].copy()
                with open('../results/pickles/{}-{}x.pickle'.format(col_name, count), 'rb') as load_model:
                    rf_model = pickle.load(load_model)

                x_data = loop_data.loc[:, features[count]].values.reshape(-1, 1)

                #Predicting using Random forest model
                y_data = rf_model.predict(x_data)

                loop_data.loc[:,col_name] = y_data

                complete_data = complete_data.append(loop_data)

            else:

                #warnings.warn("There are more than 3 missing values in one or more records")

                loop_data = all_data.loc[item].copy()

                with open('../results/pickles/fixed_values2.pickle'.format(col_name, count), 'rb') as load_model:
                    fixed_values = pickle.load(load_model)

                # Imputing with preset values as more than 4 values are missing.
                y_data = fixed_values[col_name]
                loop_data.loc[:,col_name] = y_data
                complete_data = complete_data.append(loop_data)

        all_data = complete_data.copy()

    all_data = all_data.sort_index()
    
    #Binarizing variables
    lb = LabelBinarizer()
    for col in all_data:
        target = all_data[col]
        if(len(target.unique()) <= 7):
            binary_data = lb.fit_transform(target)
            unique_vals = all_data[col].unique()
            unique_vals.sort()
            col_names = [col + '_' +str(int(item)) for item in unique_vals if item is not np.nan]
            all_data = pd.concat([all_data, pd.DataFrame(binary_data, columns= col_names)], axis = 1)
            all_data.drop([col], axis = 1, inplace = True)
    
    return all_data