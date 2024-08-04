'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC

def decision_tree(df_arrests):
    """
    Perform decision tree classification and return test set predictions.

    :param df_arrests: DataFrame containing the features and target variable
    :return: Tuple (df_arrests_test, gs_cv_dt) with predictions
    """
    # Define features and target variable
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    X = df_arrests[features]
    y = df_arrests['y']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)
    
    # Define parameter grid for decision tree
    param_grid_dt = {'max_depth': [3, 5, 7]}
    
    # Initialize Decision Tree model
    dt_model = DTC()
    
    # Initialize GridSearchCV
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='roc_auc')
    
    # Fit the model
    gs_cv_dt.fit(X_train, y_train)
    
    # Optimal value for max_depth
    best_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Optimal value for max_depth: {best_max_depth}")
    
    # Predict for the test set
    df_arrests_test = X_test.copy()
    df_arrests_test['y'] = y_test
    df_arrests_test['pred_dt'] = gs_cv_dt.predict_proba(X_test)[:, 1]
    
    return df_arrests_test, gs_cv_dt

if __name__ == "__main__":
    data_dir = "./data"
    df_arrests = pd.read_csv(f"{data_dir}/df_arrests.csv")
    df_arrests_test_with_dt, gs_cv_dt = decision_tree(df_arrests)
    df_arrests_test_with_dt.to_csv(f"{data_dir}/df_arrests_test_with_dt.csv", index=False)
