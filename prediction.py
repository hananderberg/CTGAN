import os
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LinearRegression

from datasets import datasets
import matplotlib.pyplot as plt

all_datasets = ["letter", "news", "mushroom", "credit", "bank"]
all_missingness = [10, 30, 50]
all_extra_amount = [0, 50, 100]
#all_extra_amount = [0, 50, 100]

def linearRegression(X_train, X_test, y_train, y_test):
    # Create a LinearRegression object
    lr = LinearRegression()

    # Fit the model to the training data
    lr.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lr.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return mse

def kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test, data_name):
   # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Find best k value
    k_values = [i for i in range (1,31)]
    scores = []

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=3) # three splits were used for list_wise_deletion since too few rows
        scores.append(np.mean(score))

    best_index = np.argmax(scores)
    best_k = k_values[best_index]

    # Create classifier
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Convert string targets to binary form
    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)
    y_pred_binary = lb.transform(y_pred)

    # Evaluate 
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    class_case = datasets[data_name]['classification']['class-case']
    
    if class_case == 'binary':
        auroc = roc_auc_score(y_test_binary, y_pred_binary)
    elif class_case == 'multiclass':
        auroc = roc_auc_score(y_test_binary, y_pred_binary, multi_class='ovr', average='macro')

    return accuracy, auroc

def main():
    results = []

    for extra_amount in all_extra_amount:
        for data_name in all_datasets:
            for miss_rate in all_missingness:
                filename_original_data = 'preprocessed_data/one_hot_test_data/one_hot_{}_test.csv'.format(data_name)
                original_data = pd.read_csv(filename_original_data)
                
                if extra_amount == 0:
                    filename_imputed_data = 'imputed_data/{}_{}_wo_target.csv'.format(data_name, miss_rate)
                else:
                    filename_imputed_data = 'imputed_data/{}_{}_wo_target_extra_{}.csv'.format(data_name, miss_rate, extra_amount)
                
                # Check if the file exists
                if not os.path.isfile(filename_imputed_data):
                    continue
            
                imputed_data_wo_target = pd.read_csv(filename_imputed_data)

                # Split the data into features (X) and target (y)
                X = imputed_data_wo_target
                y = original_data[datasets[data_name]["target"]]

                # Split the data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                if datasets[data_name]['classification']['model'] == KNeighborsClassifier:
                    accuracy, auroc = kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test, data_name)
                    results.append({'dataset': data_name + str(miss_rate), 'extra amount': str(extra_amount), 'scores':{'accuracy': str(accuracy), 'auroc': str(auroc)}})
                elif datasets[data_name]['classification']['model'] == LinearRegression:
                    mse = linearRegression(X_train, X_test, y_train, y_test)
                    results.append({'dataset': data_name + str(miss_rate), 'extra amount': str(extra_amount), 'scores':{'mse': str(mse)}})
            
    return results

if __name__ == '__main__':   
  results = main()
  for item in results:
    print('Dataset:', item['dataset'])
    print('Extra amount:', item['extra amount'])
    print('Scores:', item['scores'])