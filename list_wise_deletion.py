import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

from datasets import datasets

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

    no, dim = X_test.shape

    k_values = [i for i in range (1, no)]

    # Find best k value
    k_values = [i for i in range (1, no)]
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

# 1. Delete incomplete rows
def deleteIncompleteRows():
    for dataset in all_datasets:
        for miss_rate in all_missingness:
            filename = '{}_{}_{}_{}.csv'.format('preprocessed_data/one_hot_test_data/one_hot', dataset, 'test', miss_rate)
            data = pd.read_csv(filename)

            # Filter out rows with missing values and save to a new DataFrame
            data_no_nans = data.dropna()

            if data_no_nans.empty == True:
                printstatement = '{}_{}_{}'.format(dataset, miss_rate, 'data contains no full rows')
                print(printstatement)
                break
            
            filename = '{}_{}_{}_{}_{}.csv'.format('list_wise_deletion/one_hot', dataset, 'test', miss_rate, 'only_complete_rows.csv')
            data_no_nans.to_csv(filename, index=False)

def prediction():
    results = []
    
    for dataset in all_datasets:
        for miss_rate in all_missingness:
            filename = '{}_{}_{}_{}_{}.csv'.format('list_wise_deletion/one_hot', dataset, 'test', miss_rate, 'only_complete_rows.csv')

            # Check if the file exists
            if not os.path.isfile(filename):
                continue

            data = pd.read_csv(filename)

            if data.shape[0] == 1:
                continue

            # Split the data into features (X) and target (y)
            X = data.drop(data.columns[-1], axis=1)
            y = data.iloc[:, -1]

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if datasets[dataset]['classification']['model'] == KNeighborsClassifier:
                accuracy, auroc = kNeighborsClassifier(X, y, X_train, X_test, y_train, y_test, dataset)
                results.append({'Dataset': dataset, 'Missing%': str(miss_rate), 'Accuracy': str(accuracy), 'AUROC': str(auroc), 'MSE': "-"})
            elif datasets[dataset]['classification']['model'] == LinearRegression:
                mse = linearRegression(X_train, X_test, y_train, y_test)
                results.append({'Dataset': dataset, 'Missing%': str(miss_rate), 'Accuracy': "-", 'AUROC': "-", 'MSE': str(mse)})

    return results


if __name__ == '__main__': 
    all_datasets = ["mushroom", "letter", "bank", "credit", "news"]
    all_missingness = [10, 30, 50]

    #deleteIncompleteRows()
    results = prediction()
    df_results = pd.DataFrame(results, columns=['Dataset', 'Missing%', 'Accuracy', 'AUROC', 'MSE'])
    df_results.to_csv('results/list_wise_deletion_prediction.csv', index = False)

