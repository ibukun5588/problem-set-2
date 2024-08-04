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
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC

# Call functions / instantiate objects from the .py files
def main():
    """
    Main function to run the whole problem set.
    """

    # PART 1: Instantiate etl, saving the two datasets in `./data/`
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load the data
    pred_universe_raw, arrest_events_raw = etl.load_data()
    
    # Preprocess the data
    pred_universe, arrest_events = etl.preprocess_data(pred_universe_raw, arrest_events_raw)
    
    # Save the preprocessed data
    etl.save_data(pred_universe, arrest_events, data_dir)
    

    # PART 2: Call functions/instantiate objects from preprocessing
    df_arrests = preprocessing.preprocess(data_dir)

    print(df_arrests.columns)
    
    # PART 3: Call functions/instantiate objects from logistic_regression
    df_arrests_test_lr, gs_cv = logistic_regression.logistic_regression(df_arrests)
    df_arrests_test_lr.to_csv(os.path.join(data_dir, 'df_arrests_test_lr.csv'), index=False)

    # Print the optimal value for C and discuss regularization
    best_C = gs_cv.best_params_['C']
    print(f"Optimal value for C: {best_C}")
    if best_C == 0.1:
        regularization_desc = "most regularization"
    elif best_C == 10:
        regularization_desc = "least regularization"
    else:
        regularization_desc = "in the middle"
    print(f"The optimal value for C had {regularization_desc}.")
    
    # PART 4: Call functions/instantiate objects from decision_tree
    df_arrests_test_dt, gs_cv_dt = decision_tree.decision_tree(df_arrests)
    df_arrests_test_dt.to_csv(os.path.join(data_dir, 'df_arrests_test_dt.csv'), index=False)
    
    # Print the optimal value for max_depth and discuss regularization
    best_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Optimal value for max_depth: {best_max_depth}")
    if best_max_depth == 3:
        regularization_desc = "most regularization"
    elif best_max_depth == 7:
        regularization_desc = "least regularization"
    else:
        regularization_desc = "in the middle"
    print(f"The optimal value for max_depth had {regularization_desc}.")
    
    # Merge logistic regression and decision tree predictions
    df_arrests_test_with_dt = df_arrests_test_dt.copy()
    df_arrests_test_with_dt['pred_lr'] = df_arrests_test_lr['pred_lr']
    
    # Save the merged DataFrame
    df_arrests_test_with_dt.to_csv(os.path.join(data_dir, 'df_arrests_test_with_dt.csv'), index=False)

    # PART 5: Call functions/instantiate objects from calibration_plot
    print("Creating calibration plot for logistic regression model...")
    calibration_plot.calibration_plot(df_arrests_test_with_dt['y'], df_arrests_test_with_dt['pred_lr'], n_bins=5)
    
    print("Creating calibration plot for decision tree model...")
    calibration_plot.calibration_plot(df_arrests_test_with_dt['y'], df_arrests_test_with_dt['pred_dt'], n_bins=5)
    
    # Determine which model is more calibrated
    print("Which model is more calibrated? Compare the calibration plots to answer this question.")
    
    # Extra Credit
    ppv_lr, auc_lr = calibration_plot.compute_metrics(df_arrests_test_with_dt, 'pred_lr')
    ppv_dt, auc_dt = calibration_plot.compute_metrics(df_arrests_test_with_dt, 'pred_dt')
    
    print(f"PPV for logistic regression model for top 50 predicted risks: {ppv_lr}")
    print(f"PPV for decision tree model for top 50 predicted risks: {ppv_dt}")
    print(f"AUC for logistic regression model: {auc_lr}")
    print(f"AUC for decision tree model: {auc_dt}")
    
    # Print answers based on the metrics
    if auc_lr > auc_dt and ppv_lr > ppv_dt:
        model_accuracy = "Both metrics agree that the logistic regression model is more accurate."
    elif auc_dt > auc_lr and ppv_dt > ppv_lr:
        model_accuracy = "Both metrics agree that the decision tree model is more accurate."
    else:
        model_accuracy = "The metrics do not agree on which model is more accurate."
    print(model_accuracy)

if __name__ == "__main__":
    main()




