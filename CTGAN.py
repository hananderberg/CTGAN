import pandas as pd
import csv
import matplotlib.pyplot as plt
from sdv import load_demo
from sdv.tabular import CTGAN
from table_evaluator import TableEvaluator

def syntheziseData(data, batch_size, training_epochs, number_of_rows):
    ctgan = CTGAN(batch_size = batch_size, epochs = training_epochs, verbose=True)
    ctgan.fit(data)
    #CTGAN.sample(num_rows, randomize_samples=True, max_tries_per_batch=100, batch_size=None, output_file_path=None, conditions=None)
    new_data = ctgan.sample(num_rows = number_of_rows)
    return new_data
    
def evaluateSynthezisedData(data, new_data):
    table_evaluator = TableEvaluator(data, new_data)
    table_evaluator.visual_evaluation()

def saveSynthezisedDataCSV(new_data, output_file_path):
    new_data.to_csv(output_file_path, index=False)

def concatinateDatasets(data, new_data, output_file_path_merged):
    result = pd.concat([data, new_data])
    result.to_csv(output_file_path_merged, index=False)

## Read in dataset
data = pd.read_csv("Data/Customers.csv")
print(data)

## Set parameters
batch_size = 5000
training_epochs = 20
number_of_rows = 200
output_file_path = 'Data/extended_customers_dataset.csv'
output_file_path_merged = 'Data/merged_customers_dataset.csv'

new_data = syntheziseData(data, batch_size, training_epochs, number_of_rows)
saveSynthezisedDataCSV(new_data, output_file_path)
concatinateDatasets(data, new_data, output_file_path_merged)
