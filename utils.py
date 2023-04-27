import pandas as pd

    

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
    if extra_amount==50 or extra_amount==100 or extra_amount==0:
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