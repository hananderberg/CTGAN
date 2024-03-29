import re
import pandas as pd
import numpy as np



def get_values_and_bounds(df, dataset, miss_rate, evaluation, ctgan_option):
    values = get_filtered_values_df_confidence_interval(df, dataset=dataset, miss_rate=miss_rate, evaluation=evaluation, ctgan_option=ctgan_option).values.ravel()
    numbers = re.findall(r'\d+\.\d+', str(values))
    if len(numbers) == 0:
        return None, None
    lower = float(numbers[0])
    upper = float(numbers[1])
    return lower, upper

def find_smallest_value_matrix(arr_list):
    min_val = float("inf")

    # Loop through each array in the list
    for arr in arr_list:
        # Loop through each element in the array and update the minimum non-zero value if necessary
        for val in arr:
            if val != 0 and val < min_val:
                min_val = val
    return min_val

def readDataConfidenceInterval(imputation_method):
    filename = 'results/Results - '+ imputation_method + ' Confidence Interval.csv'
    df = pd.read_csv(filename, header=None, thousands=',').drop(0)
    df = df.replace(',', '', regex=True)
    df.iloc[0] = df.iloc[0].ffill()
    df.iloc[1] = df.iloc[1].ffill()
    df.iloc[:,0] = df.iloc[:,0].ffill()
    df.iloc[0,0] = "Dataset"
    df.iloc[1,0] = "Dataset"
    df.iloc[0,1] = "Missing %"
    df.iloc[1,1] = "Missing %"
    df = df.replace('-', 0)
    df.columns = pd.MultiIndex.from_arrays(df[:3].values)
    df = df[3:]
    return df

def readDataSeparateCsvBothImputationAndPrediction():
    filenames = ["results/Results - Imputation without CTGAN.csv", "results/Results - Imputation with CTGAN.csv"]
    imputation_data_frames = []
    for filename in filenames:
        df = pd.read_csv(filename, header=None, thousands=',').drop(0)
        df = df.replace(',', '', regex=True)
        df.iloc[0] = df.iloc[0].ffill()
        df.iloc[:,0] = df.iloc[:,0].ffill()
        df.iloc[0,0] = "Dataset"
        df.iloc[0,1] = "Missing %"
        df = df.replace('-', 0)
        df.columns = pd.MultiIndex.from_arrays(df[:2].values)
        df = df[2:]
        imputation_data_frames.append(df)

    # locate the rows "X% increased data"
    idx_100 = 10
    idx_200 = 20
    idx_500 = 30

    # split the DataFrame at the located row
    imputation_df_ctgan50 = imputation_data_frames[1].iloc[1:idx_100].reset_index(drop=True)
    imputation_df_ctgan100 = imputation_data_frames[1].iloc[idx_100+1:idx_200].reset_index(drop=True)
    imputation_df_ctgan200 = imputation_data_frames[1].iloc[idx_200+1:idx_500].reset_index(drop=True)
    imputation_df_ctgan500 = imputation_data_frames[1].iloc[idx_500+1:].reset_index(drop=True)

    filenames = ["results/Results - Prediction without CTGAN.csv", "results/Results - Prediction with CTGAN.csv"]
    prediction_data_frames = []
    for filename in filenames:
        df = pd.read_csv(filename, header=None, thousands=',').drop(0)
        df = df.replace(',', '', regex=True)
        df.iloc[0] = df.iloc[0].ffill()
        df.iloc[:,0] = df.iloc[:,0].ffill()
        df.iloc[0,0] = "Dataset"
        df.iloc[0,1] = "Missing %"
        df = df.replace('-', 0)
        df.columns = pd.MultiIndex.from_arrays(df[:2].values)
        df = df[2:]
        prediction_data_frames.append(df)


    # split the DataFrame at the located row
    prediction_df_ctgan50 = prediction_data_frames[1].iloc[1:idx_100].reset_index(drop=True)
    prediction_df_ctgan100 = prediction_data_frames[1].iloc[idx_100+1:idx_200].reset_index(drop=True)
    prediction_df_ctgan200 = prediction_data_frames[1].iloc[idx_200+1:idx_500].reset_index(drop=True)
    prediction_df_ctgan500 = prediction_data_frames[1].iloc[idx_500+1:].reset_index(drop=True)

    return imputation_data_frames[0], imputation_df_ctgan50, imputation_df_ctgan100, imputation_df_ctgan200, imputation_df_ctgan500, prediction_data_frames[0], prediction_df_ctgan50, prediction_df_ctgan100, prediction_df_ctgan200, prediction_df_ctgan500

def readDataSummary(args):
    if args.evaluation_type == "Prediction":
      filename = "results/Results - Prediction summary.csv"
    else:
      filename = "results/Results - Imputation summary.csv"
    df = pd.read_csv(filename, header=None, thousands=',').drop(0)
    df = df.replace(',', '', regex=True)
    df.iloc[0] = df.iloc[0].ffill()
    df.iloc[:,0] = df.iloc[:,0].ffill()
    df.iloc[:,1] = df.iloc[:,1].ffill()
    df.iloc[0,0] = "Dataset"
    df.iloc[0,1] = "Missing %"
    df.iloc[0,2] = "Additional CTGAN data%"
    df = df.replace('-', 0)
    df.columns = pd.MultiIndex.from_arrays(df[:2].values)
    df = df[2:]
    #df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    return df

def readDataSeparateCsv(args):
    # read the CSV files and drop the first row
    if args.evaluation_type == "Prediction":
      filenames = ["results/Results - Prediction without CTGAN.csv", "results/Results - Prediction with CTGAN.csv"]
    else:
      filenames = ["results/Results - Imputation without CTGAN.csv", "results/Results - Imputation with CTGAN.csv"]
    data_frames = []
    for filename in filenames:
        df = pd.read_csv(filename, header=None, thousands=',').drop(0)
        df = df.replace(',', '', regex=True)
        df.iloc[0] = df.iloc[0].ffill()
        df.iloc[:,0] = df.iloc[:,0].ffill()
        df.iloc[0,0] = "Dataset"
        df.iloc[0,1] = "Missing %"
        df = df.replace('-', 0)
        df.columns = pd.MultiIndex.from_arrays(df[:2].values)
        df = df[2:]
        data_frames.append(df)

    # locate the row "100% increased data"
    idx = 10

    # split the DataFrame at the located row
    df_ctgan50 = data_frames[1].iloc[1:idx].reset_index(drop=True)
    df_ctgan100 = data_frames[1].iloc[idx+1:].reset_index(drop=True)

    return data_frames[0], df_ctgan50, df_ctgan100

def find_best_value_stat_sign(gainv1_lower_0, gainv1_upper_0, gainv1_lower_ctgan, gainv1_upper_ctgan, evaluation):
    if evaluation == "Accuracy" or evaluation == "AUROC": # Max value is better
        if gainv1_upper_0 > gainv1_upper_ctgan and gainv1_lower_0 > gainv1_upper_ctgan: # 0 % is stat sign better
            return 1
        elif gainv1_upper_ctgan > gainv1_upper_0 and gainv1_lower_ctgan > gainv1_upper_0: # ctgan is stat sign better
            return 2
        else: 
            return 3
    else: # Min value is better
        if gainv1_lower_0 < gainv1_lower_ctgan and gainv1_upper_0 < gainv1_lower_ctgan: # 0 % is stat sign better
            return 1
        elif gainv1_lower_ctgan < gainv1_lower_0 and gainv1_lower_ctgan < gainv1_upper_0: # 50 % is stat sign better
            return 2
        else: 
            return 3

def find_best_value(value1, value2, evaluation):
    if evaluation == "Accuracy" or evaluation == "AUROC": # Max value is better
        if value1 == value2:
            return 3 # It is a tie
        elif value1 > value2:
            return 1
        else:
            return 2
    else:  # Min value is better
        if value1 == value2:
            return 3 # It is a tie
        elif value1 < value2:
            return 1
        else:
            return 2
    
def find_evaluation_type(evaluation_type, imputation_evaluation, prediction_evaluation):
    if evaluation_type == "Prediction":
       return prediction_evaluation
    elif evaluation_type == "Imputation":
       return imputation_evaluation
    
    return None

def find_no_training_samples(ctgan_option, dataset):
    if dataset == "mushroom":
        no_training = 6499
        if ctgan_option == "CTGAN 50%": 
            return no_training + no_training/2
        elif ctgan_option == "CTGAN 100%":
            return no_training + no_training
        else:
            return no_training  
    elif dataset == "letter":
        no_training = 16000
        if ctgan_option == "CTGAN 50%": 
            return no_training + no_training/2
        elif ctgan_option == "CTGAN 100%":
            return no_training + no_training
        else:
            return no_training  
    elif dataset == "bank":
        no_training = 32950
        if ctgan_option == "CTGAN 50%": 
            return no_training + no_training/2
        elif ctgan_option == "CTGAN 100%":
            return no_training + no_training
        else:
            return no_training  
    elif dataset == "credit":
        no_training = 24000

        if ctgan_option == "CTGAN 50%": 
            return no_training + no_training/2
        elif ctgan_option == "CTGAN 100%":
            return no_training + no_training
        else:
            return no_training
    elif dataset == "news":
        no_training = 31715
        if ctgan_option == "CTGAN 50%": 
            return no_training + no_training/2
        elif ctgan_option == "CTGAN 100%":
            return no_training + no_training
        else:
            return no_training 
        
def get_filtered_values_df_confidence_interval(df, dataset=None, miss_rate=None, evaluation=None, ctgan_option=None):
    if dataset:
        df = df.loc[df.iloc[:, 0].str.lower() == dataset]
    if miss_rate:
        df = df.loc[df.iloc[:, 1].astype(str) == str(miss_rate)]
    if evaluation:
        df = df.loc[:, df.columns.get_level_values(2) == evaluation]
    if ctgan_option:
        df = df.loc[:, df.columns.get_level_values(1) == ctgan_option]

    return df  
        
def get_filtered_values_df(df, dataset=None, miss_rate=None, imputation_method=None, evaluation=None):
    if dataset:
        df = df.loc[df.iloc[:, 0].str.lower() == dataset]
    if miss_rate:
        df = df.loc[df.iloc[:, 1].astype(str) == str(miss_rate)]
    if imputation_method:
        df = df.loc[:, df.columns.get_level_values(0) == imputation_method]
    if evaluation:
        df = df.loc[:, df.columns.get_level_values(1) == evaluation]

    return df  
        
def get_filtered_values(df, dataset=None, miss_rate=None, extra_amount=None, imputation_method=None, evaluation=None):
    if dataset:
        df = df.loc[df.iloc[:, 0].str.lower() == dataset]
    if miss_rate:
        df = df.loc[df.iloc[:, 1].astype(str) == str(miss_rate)]
    if extra_amount or extra_amount==0:
        df = df.loc[df.iloc[:, 2].astype(str) == str(extra_amount)]
    if imputation_method:
        df = df.loc[:, df.columns.get_level_values(0) == imputation_method]
    if evaluation:
        df = df.loc[:, df.columns.get_level_values(1) == evaluation]

    return df     

def get_filtered_values_separateCsv(df, dataset=None, miss_rate=None, imputation_method=None, evaluation=None):
    if dataset:
        df = df.loc[df.iloc[:, 0].str.lower() == dataset]
    if miss_rate:
        df = df.loc[df.iloc[:, 1].astype(str) == str(miss_rate)]
    if imputation_method:
        df = df.loc[:, df.columns.get_level_values(0) == imputation_method]
    if evaluation:
        df = df.loc[:, df.columns.get_level_values(1) == evaluation]

    values = df.values.flatten().astype(float)
    return values

def find_miss_rates(dataset_values):
    """
    Find miss rates for the current dataset_values
    """
    if len(dataset_values) == 3:
      miss_rates = [10, 30, 50] 
    elif len(dataset_values) == 2:
      miss_rates = [10, 30]
    else:
      miss_rates = [10]

    return miss_rates

def collect_handles_and_labels(bars, x_axis_options):
      handles, labels = [], []
      for j, bar in enumerate(bars):
          handles.append(bar)
          labels.append(x_axis_options[j])
      return handles, labels

def is_vector_all_zeros(arr):
    """
    Checks if a vector only contains zeros.
    
    Returns:
    True if the vector only contains zeros, False otherwise
    """
    for elem in arr:
        if elem != 0:
            return False
    return True

def is_matrix_all_zeros(values):
    """
    Checks if a matrix only contains zeros.
    
    Args:
    values: A 2D list representing the matrix
    
    Returns:
    True if the matrix only contains zeros, False otherwise
    """
    for row in values:
        for val in row:
            if float(val) != 0:
                return False
    return True


def get_all_upper_and_lower(df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2, data_set, miss_rate, evaluation):
    mf_lower_0, mf_upper_0 = get_values_and_bounds(df_confidence_interval_missforest, data_set, miss_rate, evaluation, "0 % CTGAN")
    
    if mf_lower_0 == None and mf_upper_0 == None:
        return None, None
    
    mf_lower_50, mf_upper_50 = get_values_and_bounds(df_confidence_interval_missforest, data_set, miss_rate, evaluation, "50 % CTGAN")
    mf_lower_100, mf_upper_100 = get_values_and_bounds(df_confidence_interval_missforest, data_set, miss_rate, evaluation, "100 % CTGAN")

    gainv1_lower_0, gainv1_upper_0 = get_values_and_bounds(df_confidence_interval_gainv1, data_set, miss_rate, evaluation, "0 % CTGAN")
    gainv2_lower_0, gainv2_upper_0 = get_values_and_bounds(df_confidence_interval_gainv2, data_set, miss_rate, evaluation, "0 % CTGAN")
            
    gainv1_lower_50, gainv1_upper_50 = get_values_and_bounds(df_confidence_interval_gainv1, data_set, miss_rate, evaluation, "50 % CTGAN")
    gainv2_lower_50, gainv2_upper_50 = get_values_and_bounds(df_confidence_interval_gainv2, data_set, miss_rate, evaluation, "50 % CTGAN")

    gainv1_lower_100, gainv1_upper_100 = get_values_and_bounds(df_confidence_interval_gainv1, data_set, miss_rate, evaluation, "100 % CTGAN")
    gainv2_lower_100, gainv2_upper_100 = get_values_and_bounds(df_confidence_interval_gainv2, data_set, miss_rate, evaluation, "100 % CTGAN")

    gainv1_lower_200, gainv1_upper_200 = get_values_and_bounds(df_confidence_interval_gainv1, data_set, miss_rate, evaluation, "200 % CTGAN")
    gainv2_lower_200, gainv2_upper_200 = get_values_and_bounds(df_confidence_interval_gainv2, data_set, miss_rate, evaluation, "200 % CTGAN")

    gainv1_lower_500, gainv1_upper_500 = get_values_and_bounds(df_confidence_interval_gainv1, data_set, miss_rate, evaluation, "500 % CTGAN")
    gainv2_lower_500, gainv2_upper_500 = get_values_and_bounds(df_confidence_interval_gainv2, data_set, miss_rate, evaluation, "500 % CTGAN")

    value_medianmode_0 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='Median/mode', evaluation=evaluation).values.ravel()[0])
    value_MICE_0 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='MICE', evaluation=evaluation).values.ravel()[0])
    value_kNN_0 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='kNN', evaluation=evaluation).values.ravel()[0])
            
    value_medianmode_50 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=50, imputation_method='Median/mode', evaluation=evaluation).values.ravel()[0])
    value_MICE_50 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=50, imputation_method='MICE', evaluation=evaluation).values.ravel()[0])
    value_kNN_50 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=50, imputation_method='kNN', evaluation=evaluation).values.ravel()[0])
            
    value_medianmode_100 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=100, imputation_method='Median/mode', evaluation=evaluation).values.ravel()[0])
    value_MICE_100 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=100, imputation_method='MICE', evaluation=evaluation).values.ravel()[0])
    value_kNN_100 = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=100, imputation_method='kNN', evaluation=evaluation).values.ravel()[0])
    
    if evaluation in ['Accuracy', 'AUROC', 'MSE']: # It is prediction
        value_listwise = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=100, imputation_method='List-wise deletion', evaluation=evaluation).values.ravel()[0])
        if value_listwise == 0:
            if evaluation == "MSE":
                value_listwise = np.inf
            else:
                value_listwise = -np.inf
        all_lower = [value_medianmode_0, value_medianmode_50, value_medianmode_100, value_MICE_0, value_MICE_50, value_MICE_100, value_kNN_0, value_kNN_50, value_kNN_100, mf_lower_0, mf_lower_50, mf_lower_100, \
                            gainv1_lower_0, gainv1_lower_50, gainv1_lower_100, gainv1_lower_200, gainv1_lower_500, gainv2_lower_0, gainv2_lower_50, gainv2_lower_100, gainv2_lower_200, gainv2_lower_500]
        all_upper = [value_medianmode_0, value_medianmode_50, value_medianmode_100, value_MICE_0, value_MICE_50, value_MICE_100, value_kNN_0, value_kNN_50, value_kNN_100, mf_upper_0, mf_upper_50, mf_upper_100, \
                         gainv1_upper_0, gainv1_upper_50, gainv1_upper_100, gainv1_upper_200, gainv1_upper_500, gainv2_upper_0, gainv2_upper_50, gainv2_upper_100, gainv2_upper_200, gainv2_upper_500]
    else:
        all_lower = [value_medianmode_0, value_medianmode_50, value_medianmode_100, value_MICE_0, value_MICE_50, value_MICE_100, value_kNN_0, value_kNN_50, value_kNN_100, mf_lower_0, mf_lower_50, mf_lower_100, \
                            gainv1_lower_0, gainv1_lower_50, gainv1_lower_100, gainv1_lower_200, gainv1_lower_500, gainv2_lower_0, gainv2_lower_50, gainv2_lower_100, gainv2_lower_200, gainv2_lower_500]
        all_upper = [value_medianmode_0, value_medianmode_50, value_medianmode_100, value_MICE_0, value_MICE_50, value_MICE_100, value_kNN_0, value_kNN_50, value_kNN_100, mf_upper_0, mf_upper_50, mf_upper_100, \
                         gainv1_upper_0, gainv1_upper_50, gainv1_upper_100, gainv1_upper_200, gainv1_upper_500, gainv2_upper_0, gainv2_upper_50, gainv2_upper_100, gainv2_upper_200, gainv2_upper_500]
              
    return all_lower, all_upper