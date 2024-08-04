import pandas as pd
import numpy as np
import os
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    df_arrests = pd.read_csv(os.path.join(data_dir, 'df_arrests.csv'))
    df_arrests_test, gs_cv_dt = decision_tree(df_arrests)
    df_arrests_test.to_csv(os.path.join(data_dir, 'df_arrests_test_dt.csv'), index=False)
