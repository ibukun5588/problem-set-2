'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''
import pandas as pd

def load_data(data_dir):
    """
    Load the preprocessed datasets from the data directory.
    
    :param data_dir: Directory where the datasets are
    :return: Tuple of DataFrames (pred_universe, arrest_events)
    """
    pred_universe_path = f"{data_dir}/pred_universe_raw.csv"
    arrest_events_path = f"{data_dir}/arrest_events_raw.csv"
    
    pred_universe = pd.read_csv(pred_universe_path)
    arrest_events = pd.read_csv(arrest_events_path)
    
    return pred_universe, arrest_events

def merge_datasets(pred_universe, arrest_events):
    """
    Perform a full outer join/merge on 'person_id' into `df_arrests`.
    
    :param pred_universe: DataFrame of the pred_universe dataset
    :param arrest_events: DataFrame of the arrest_events dataset
    :return: Merged DataFrame `df_arrests`
    """
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')
    return df_arrests

def create_target_variable(df_arrests):
    """
    Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime 
    in the 365 days after their arrest date.
    
    :param df_arrests: Merged DataFrame of arrests
    :return: DataFrame with the target variable `y`
    """
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'])
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
    
    def check_rearrest(row):
        future_arrests = df_arrests[
            (df_arrests['person_id'] == row['person_id']) & 
            (df_arrests['arrest_date_event'] > row['arrest_date_univ']) & 
            (df_arrests['arrest_date_event'] <= row['arrest_date_univ'] + pd.DateOffset(days=365)) & 
            (df_arrests['charge_degree'] == 'felony')
        ]
        return 1 if not future_arrests.empty else 0

    df_arrests['y'] = df_arrests.apply(check_rearrest, axis=1)
    return df_arrests

def share_rearrested_felony(df_arrests):
    """
    Calculate the share of arrestees from `df_arrests` who were rearrested within a year of their release.
    
    :param df_arrests: DataFrame with the target variable `y`
    :return: Share of rearrested arrestees for a felony crime
    """
    share = df_arrests['y'].mean()
    print(f"What share of arrestees were rearrested for a felony crime in the next year? {share * 100:.2f}%")
    return share

def create_current_charge_felony(df_arrests):
    """
    Create a predictive feature `current_charge_felony` which equals 1 if the current arrest was for a felony charge, 
    and 0 otherwise.
    
    :param df_arrests: DataFrame of arrests
    :return: DataFrame with the predictive feature `current_charge_felony`
    """
    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'felony' else 0)
    share = df_arrests['current_charge_felony'].mean()
    print(f"What share of current charges are felonies? {share * 100:.2f}%")
    return df_arrests

def create_num_fel_arrests_last_year(df_arrests):
    """
    Create a predictive feature `num_fel_arrests_last_year` which is the total number of arrests in the one year 
    prior.
    
    :param df_arrests: DataFrame of arrests
    :return: DataFrame with the predictive feature `num_fel_arrests_last_year`
    """
    def count_past_arrests(row):
        past_arrests = df_arrests[
            (df_arrests['person_id'] == row['person_id']) & 
            (df_arrests['arrest_date_event'] >= row['arrest_date_univ'] - pd.DateOffset(days=365)) & 
            (df_arrests['arrest_date_event'] < row['arrest_date_univ']) & 
            (df_arrests['charge_degree'] == 'felony')
        ]
        return len(past_arrests)

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(count_past_arrests, axis=1)
    average_num_fel_arrests = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {average_num_fel_arrests:.2f}")
    return df_arrests

def preprocess(data_dir):
    """
    Perform all preprocessing steps together all at once.
    
    :param data_dir: Directory where the datasets are stored
    :return: Preprocessed DataFrame `df_arrests`
    """
    # Load data
    pred_universe, arrest_events = load_data(data_dir)
    
    # Merge datasets
    df_arrests = merge_datasets(pred_universe, arrest_events)
    
    # Create target variable `y`
    df_arrests = create_target_variable(df_arrests)
    share_rearrested_felony(df_arrests)
    
    # Create predictive feature `current_charge_felony`
    df_arrests = create_current_charge_felony(df_arrests)
    
    # Create predictive feature `num_fel_arrests_last_year`
    df_arrests = create_num_fel_arrests_last_year(df_arrests)
    
    # Print the mean of 'num_fel_arrests_last_year'
    print(df_arrests['num_fel_arrests_last_year'].mean())
    
    # Print the head of the dataframe
    print(df_arrests.head())
    
    return df_arrests

if __name__ == "__main__":
    data_dir = "./data"
    df_arrests = preprocess(data_dir)
    df_arrests.to_csv(f"{data_dir}/df_arrests.csv", index=False)
