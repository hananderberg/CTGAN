import pandas as pd
import numpy as np

from datasets import datasets
import matplotlib.pyplot as plt
from sdv import load_demo
from sdv.tabular import CTGAN
from table_evaluator import TableEvaluator

all_datasets = ["mushroom", "credit", "letter", "bank", "news"]
#all_datasets = ["news"]
all_missingness = [10, 30, 50]

## Set parameters
batch_size = 500
training_epochs = 300
pac = 10

def main():
    for data_name in all_datasets:
        for missingness in all_missingness:
            filename = '{}{}_{}_{}.csv'.format('train_test_split_data/train_data_wo_target/', data_name, 'train', missingness)
            data = pd.read_csv(filename)

            filename = '{}{}_{}.csv'.format('train_test_split_data/train_data_wo_target/', data_name, 'train')
            data_no_missingness = pd.read_csv(filename)

            no, dim = data.shape
            #number_of_rows_50 = round(no/2)
            #number_of_rows_100 = no
            number_of_rows_200 = no*2
            number_of_rows_500 = no*5

            # Filter out rows with missing values and save to a new DataFrame
            data_no_nans = data.dropna()

            if data_no_nans.empty == True:
                printstatement = '{}_{}_{}'.format(data_name, missingness, 'data contains no full rows')
                print(printstatement)
                break

            ctgan = CTGAN(batch_size = batch_size, epochs = training_epochs, pac = pac, verbose=True)
            ctgan.fit(data_no_nans)

            # Generate data
            new_data_200 = ctgan.sample(num_rows = number_of_rows_200)
            new_data_500 = ctgan.sample(num_rows = number_of_rows_500)

            new_data_200_copy = new_data_200.copy()
            new_data_500_copy = new_data_500.copy()

            # Select random rows for insertion
            insert_idx_200 = np.random.choice(len(new_data_200), size=len(data), replace=False)
            insert_idx_500 = np.random.choice(len(new_data_500), size=len(data), replace=False)
            
            # Insert new rows
            for i, idx in enumerate(insert_idx_200):
                new_data_200 = pd.concat([new_data_200.iloc[:idx+i], data.iloc[[i]], new_data_200.iloc[idx+i:]], ignore_index=True)
                new_data_200_copy = pd.concat([new_data_200_copy.iloc[:idx+i], data_no_missingness.iloc[[i]], new_data_200_copy.iloc[idx+i:]], ignore_index=True)
            
            for i, idx in enumerate(insert_idx_500):
                new_data_500 = pd.concat([new_data_500.iloc[:idx+i], data.iloc[[i]], new_data_500.iloc[idx+i:]], ignore_index=True)
                new_data_500_copy = pd.concat([new_data_500_copy.iloc[:idx+i], data_no_missingness.iloc[[i]], new_data_500_copy.iloc[idx+i:]], ignore_index=True)

            # Save 200% new data, with and without missingess
            filename = '{}{}_{}_{}_{}.csv'.format('train_test_split_data/train_data_wo_target_extra_200/', data_name, 'train', missingness, 'extra_200')
            new_data_200.to_csv(filename, index=False)
            filename = '{}{}_{}{}_{}.csv'.format('train_test_split_data/train_data_wo_target_extra_200/', data_name, 'train_full', missingness, 'extra_200')
            new_data_200_copy.to_csv(filename, index=False)

            # Save 500% new data, with and without missingess
            filename = '{}{}_{}_{}_{}.csv'.format('train_test_split_data/train_data_wo_target_extra_500/', data_name, 'train', missingness, 'extra_500')
            new_data_500.to_csv(filename, index=False)
            filename = '{}{}_{}{}_{}.csv'.format('train_test_split_data/train_data_wo_target_extra_500/', data_name, 'train_full', missingness,'extra_500')
            new_data_500_copy.to_csv(filename, index=False)

            print(no)
            print(number_of_rows_200)
            print(number_of_rows_500)

            print(new_data_500.shape)
            print(new_data_500_copy.shape)

            print(new_data_500.iloc[3, 2])
            print(new_data_500_copy.iloc[3, 2])

            print(new_data_200.shape)
            print(new_data_200_copy.shape)

if __name__ == '__main__':   
    main()

    


    

