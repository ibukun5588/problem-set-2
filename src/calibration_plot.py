'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    # Calculate calibration values
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_pred, prob_true, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def compute_metrics(df, pred_col, true_col='y'):
    """
    Compute metrics for the model predictions.

    Parameters:
        df (DataFrame): DataFrame containing the true labels and predictions.
        pred_col (str): Column name for the predicted probabilities.
        true_col (str): Column name for the true labels.

    Returns:
        ppv_top_50 (float): Positive Predictive Value for the top 50 predicted risks.
        auc (float): Area Under the ROC Curve.
    """
    # Compute PPV for the top 50 predicted risks
    df_sorted = df.sort_values(by=pred_col, ascending=False)
    top_50 = df_sorted.head(50)
    ppv_top_50 = top_50[true_col].mean()
    
    # Compute AUC
    auc = roc_auc_score(df[true_col], df[pred_col])
    
    return ppv_top_50, auc

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    # Load data
    df_arrests_test_with_dt = pd.read_csv(os.path.join(data_dir, 'df_arrests_test_with_dt.csv'))
    
    # Calibration plots
    print("Creating calibration plot for logistic regression model...")
    calibration_plot(df_arrests_test_with_dt['y'], df_arrests_test_with_dt['pred_lr'], n_bins=5)
    
    print("Creating calibration plot for decision tree model...")
    calibration_plot(df_arrests_test_with_dt['y'], df_arrests_test_with_dt['pred_dt'], n_bins=5)
    
    # Determine which model is more calibrated
    # This is a subjective assessment based on the calibration plots
    print("Which model is more calibrated? Compare the calibration plots to answer this question.")
    
    # Extra Credit
    ppv_lr, auc_lr = compute_metrics(df_arrests_test_with_dt, 'pred_lr')
    ppv_dt, auc_dt = compute_metrics(df_arrests_test_with_dt, 'pred_dt')
    
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


