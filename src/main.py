'''
You will run this problem set from main.py, so set things up accordingly
'''
import pandas as pd
import os
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot

"""
import calibration_plot
"""

'''
You will run this problem set from main.py, so set things up accordingly
'''

# Call functions / instantiate objects from the .py files
def main():
    """
    Main function to run the whole problem set.
    """

    # PART 1: Instantiate etl, saving the two datasets in `./data/`
    data_dir = "./data"
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
    df_arrests_test_lr.to_csv(f"{data_dir}/df_arrests_test_lr.csv", index=False)

    # PART 4: Call functions/instantiate objects from decision_tree
    df_arrests_test_dt, gs_cv_dt = decision_tree.decision_tree(df_arrests)
    df_arrests_test_dt.to_csv(f"{data_dir}/df_arrests_test_dt.csv", index=False)
    
    # Merge logistic regression and decision tree predictions
    df_arrests_test_with_dt = df_arrests_test_dt.copy()
    df_arrests_test_with_dt['pred_lr'] = df_arrests_test_lr['pred_lr']
    
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
    
    print("Do both metrics agree that one model is more accurate than the other?")
    if auc_lr > auc_dt and ppv_lr > ppv_dt:
        print("Both metrics agree that the logistic regression model is more accurate.")
    elif auc_dt > auc_lr and ppv_dt > ppv_lr:
        print("Both metrics agree that the decision tree model is more accurate.")
    else:
        print("The metrics do not agree on which model is more accurate.")


if __name__ == "__main__":
    main()
