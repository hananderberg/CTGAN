import pandas as pd
import csv
import matplotlib.pyplot as plt
from sdv import load_demo
from sdv.tabular import CTGAN
from table_evaluator import TableEvaluator

# Read in dataset
data = pd.read_csv("Data/Salary_dataset.csv")
print(data)

# Define categorical features
# categorical_features = ['']

ctgan = CTGAN(verbose=True)
#ctgan.fit(data, categorical_features, epochs=200)
ctgan.fit(data)

new_data = ctgan.sample(num_rows=200)
new_data.head()
print(new_data)

# Evaluation
print(data.shape, new_data.shape)
table_evaluator = TableEvaluator(data, new_data)
#table_evaluator = TableEvaluator(data, new_data, cat_col=categorical_features)
table_evaluator.visual_evaluation()

# Save synthethic data
new_data.to_csv('extended_salary_dataset.csv', index=False)
