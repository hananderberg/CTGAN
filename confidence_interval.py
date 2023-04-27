
import argparse
from utils import readDataSeparateCsvBothImputationAndPrediction, get_filtered_values

import numpy as np
import pandas as pd

def calcCI(mean, st_dev, samplesize, z):
    mean = float(mean)
    st_dev = float(st_dev)

    CI_lower = mean - (z * st_dev/np.sqrt(samplesize))
    CI_upper = mean + (z * st_dev/np.sqrt(samplesize))
    return CI_lower, CI_upper

def computeCI(imputation_method, imputation_df, imputation_df_ctgan50, imputation_df_ctgan100, prediction_df, prediction_df_ctgan50, prediction_df_ctgan100, samplesize, z):
    all_dfs = imputation_df, imputation_df_ctgan50, imputation_df_ctgan100, prediction_df, prediction_df_ctgan50, prediction_df_ctgan100
    list = ["imputation", "imputation50CTGAN", "imputation100CTGAN", "prediction", "prediction50CTGAN", "prediction100CTGAN"]

    for k, df in enumerate(all_dfs):
        df_values = get_filtered_values(df, dataset=None, miss_rate=None, imputation_method=imputation_method)
        str_st_dev = "Standard Deviation " + imputation_method
        df_standard_dev = get_filtered_values(df, dataset=None, miss_rate=None, imputation_method=str_st_dev)

        # Create an empty dataframe with the same shape as the original dataframes
        result_df = pd.DataFrame(columns=df_values.columns, index=df_values.index)

        # Iterate over the values of both dataframes using a nested for loop
        for i in range(len(df_values)):
            for j in range(len(df_values.columns)):
                mean = df_values.iloc[i, j]
                st_dev = df_standard_dev.iloc[i, j]
                
                # Perform some operation on the values
                CI_lower, CI_upper = calcCI(mean, st_dev, samplesize, z)
                result = f'({CI_lower}, {CI_upper})'
                
                # Save the result to the new dataframe
                result_df.iloc[i, j] = result
        
        # Save to csv
        print(result_df)
        string = list[k]
        filename = 'results/{}_CI_{}.csv'.format(args.imputation_method, string)
        result_df.to_csv(filename, index = False)

def main(args):
    #freedomdegrees = samplesize - 1
    #confidence_interval = 0.95
    #alpha = (1 - confidence_interval) / 2

    imputation_df, imputation_df_ctgan50, imputation_df_ctgan100, prediction_df, prediction_df_ctgan50, prediction_df_ctgan100 = readDataSeparateCsvBothImputationAndPrediction()
    computeCI(args.imputation_method, imputation_df, imputation_df_ctgan50, imputation_df_ctgan100, prediction_df, prediction_df_ctgan50, prediction_df_ctgan100, args.sample_size, args.z)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--imputation_method',
      choices = ['MissForest', 'GAIN v1', 'GAIN v2'],
      default='GAIN v1',
      type=str)
    parser.add_argument(
      '--sample_size',
      default=10,
      type=str)
    parser.add_argument(
      '--z',
      default=2.262,
      type=str)
    args = parser.parse_args()
    main(args)

