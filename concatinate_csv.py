import numpy as np
import pandas as pd



d2 = pd.read_csv('results/summary_letter_mush_cred_gain_v2.csv')
d1 = pd.read_csv('results/summary_news_v2.csv')



df = pd.concat([d1,d2], ignore_index=True)

# define the custom sort order for column 'Dataset'
order = ['mushroom', 'letter', 'bank', 'credit', 'news']

# create a temporary column to represent the order of column 'Dataset'
df['Dataset_order'] = pd.Categorical(df['Dataset'], categories=order, ordered=True).codes

# sort the DataFrame by columns 'Additional CTGAN data%' and 'Dataset_order'
df = df.sort_values(by=['Additional CTGAN data%', 'Dataset_order'], ascending=[True, True])

# drop the temporary column
df = df.drop('Dataset_order', axis=1)

# get the list of column names
columns = df.columns.tolist()

# get the list of column names containing "St Dev"
stdev_columns = [col for col in columns if 'St Dev' in col]

# remove the stdev_columns from the list of columns
columns = [col for col in columns if col not in stdev_columns]

# append the stdev_columns to the end of the list of columns
columns.extend(stdev_columns)

# create a new DataFrame with the columns in the new order
df = df[columns]

df = df.fillna('-')

df.to_csv('results/summary_imputation_gain_v2.csv', index=False)
print(df)