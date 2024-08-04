'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression as lr

def logistic_regression(df_arrests):
    """
    Perform logistic regression classification and return test set predictions.

    :param df_arrests: DataFrame containing the features and target variable
    :return: Tuple (df_arrests_test, gs_cv) with predictions
    """
    # Define features and target variable
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    X = df_arrests[features]
    y = df_arrests['y']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=42)
    
    # Define parameter grid for logistic regression
    param_grid = {'C': [0.1, 1, 10]}
    
    # Initialize Logistic Regression model
    lr_model = lr(solver='liblinear')
    
    # Initialize GridSearchCV
    gs_cv = GridSearchCV(lr_model, param_grid, cv=5, scoring='roc_auc')
    
    # Fit the model
    gs_cv.fit(X_train, y_train)
    
    # Optimal value for C
    best_C = gs_cv.best_params_['C']
    print(f"Optimal value for C: {best_C}")
    
    # Predict for the test set
    df_arrests_test = X_test.copy()
    df_arrests_test['y'] = y_test
    df_arrests_test['pred_lr'] = gs_cv.predict_proba(X_test)[:, 1]
    
    return df_arrests_test, gs_cv

if __name__ == "__main__":
    data_dir = "./data"
    df_arrests = pd.read_csv(f"{data_dir}/df_arrests.csv")
    df_arrests_test, gs_cv = logistic_regression(df_arrests)
    df_arrests_test.to_csv(f"{data_dir}/df_arrests_test.csv", index=False)
