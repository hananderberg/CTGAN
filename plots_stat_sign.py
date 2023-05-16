import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import brewer2mpl

import numpy as np
from utils import collect_handles_and_labels, find_best_value, find_best_value_stat_sign, find_evaluation_type, get_all_upper_and_lower, get_filtered_values, get_values_and_bounds, is_matrix_all_zeros, is_vector_all_zeros, readDataConfidenceInterval, readDataSummary, get_filtered_values_df_confidence_interval

def main(args):
    df_confidence_interval_missforest = readDataConfidenceInterval("MissForest")
    df_confidence_interval_gainv1 = readDataConfidenceInterval("GAIN v1")
    df_confidence_interval_gainv2 = readDataConfidenceInterval("GAIN v2")
    df_summary = readDataSummary(args)

    #plotBarChartNoBestResultBaselineMethodsStatSign(args, df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2)
    #plotBarChartNoBestResultAllMethodsStatSign(args, df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2)
    #plotCTGANImpactNoBestResult(args, df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2) 
    plotGAINv1toGAINv2StatSign(args, df_confidence_interval_gainv1, df_confidence_interval_gainv2)

def plotGAINv1toGAINv2StatSign(args, df_confidence_interval_gainv1, df_confidence_interval_gainv2):
    if args.evaluation_type == "Prediction":
       all_evaluation_types = args.all_prediction_evaluations
       evaluation_labels = ['Accuracy', 'AUROC', 'MSE']
    else:
       all_evaluation_types = args.all_imputation_evaluations
       evaluation_labels = ['mRMSE', 'RMSE numerical', 'RMSE categorical', 'PFC', 'Execution time']
    all_imputation_methods = ['GAIN v1', 'GAIN v2']
    all_datasets, all_miss_rates = args.all_datasets, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    all_y_values = []
    no_results_stat_sign = [0] * len(all_evaluation_types)
    no_results_not_stat_sign = [0] * len(all_evaluation_types)

    for j, evaluation_type in enumerate(all_evaluation_types):
      y_values = [0] * len(all_imputation_methods)

      for dataset in all_datasets:
        for miss_rate in all_miss_rates:
          gainv1_lower, gainv1_upper = get_values_and_bounds(df_confidence_interval_gainv1, dataset, miss_rate, evaluation_type, "0 % CTGAN")
          gainv2_lower, gainv2_upper = get_values_and_bounds(df_confidence_interval_gainv2, dataset, miss_rate, evaluation_type, "0 % CTGAN")

          if gainv1_lower == None:
            continue

          if args.evaluation_type == "Prediction" and dataset != "news": # Max is considered the best
            if gainv1_upper > gainv2_upper and gainv1_lower > gainv2_upper:
               idx = 1
            elif gainv2_upper > gainv1_upper and gainv2_lower > gainv1_upper:
               idx = 2
            else: 
               idx = 3
          elif args.evaluation_type == "Imputation" or (dataset == "news" and args.evaluation_type == "Prediction"): # Min is considered the best
            if gainv1_lower < gainv2_lower and gainv1_upper < gainv2_lower:
               idx = 1
            elif gainv2_lower < gainv1_lower and gainv2_upper < gainv1_lower:
               idx = 2
            else: 
               idx = 3
          
          if idx == 3:
            no_results_not_stat_sign[j] += 1
          else:
            y_values[(idx-1)] += 1
            no_results_stat_sign[j] += 1
      
      all_y_values.append(y_values)

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(all_evaluation_types), figsize=(11, 3.5), sharex='col') 
   
    for i, evaluation_type in enumerate(all_evaluation_types):
      #x_ticks = np.arange(len(all_imputation_methods))
      bar_width = 0.25
      bar_positions = [0.4, 1]

      ax = axes[i]
      ax.bar(bar_positions, all_y_values[i], width=bar_width, color=colors)
      
      ax.set_xticks(bar_positions)
      ax.set_xticklabels(all_imputation_methods, rotation=0, ha='center', fontsize=9)
      ax.set_title(evaluation_labels[i])
      ax.text(0.5, -0.45, "# of results sig. at 95% conf.: " + str(no_results_stat_sign[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.text(0.5, -0.55, "# of data sets in total:  " + str(no_results_not_stat_sign[i]+no_results_stat_sign[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

      ax.set_ylim(bottom=0, top=2.2)

      # Add horizontal lines for every integer on the y-axis
      for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
        ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)

    # Set titles
    #fig.suptitle('Number of best performer per evaluation metric for all baseline methods', y=0.98, fontsize=12)
    fig.text(0.03, 0.5, '# of best perfomer', va='center', rotation='vertical', fontsize=10) 

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.show()

def plotGAINv1toGAINv2(args, df_summary):
    if args.evaluation_type == "Prediction":
       all_evaluation_types = args.all_prediction_evaluations
       evaluation_labels = ['Accuracy', 'AUROC', 'MSE']
    else:
       all_evaluation_types = args.all_imputation_evaluations
       evaluation_labels = ['mRMSE', 'RMSE numerical', 'RMSE categorical', 'PFC', 'Execution time']
    all_imputation_methods = ['GAIN v1', 'GAIN v2']
    all_datasets, all_miss_rates = args.all_datasets, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    all_y_values = []
    nr_datasets_in_evaluation = [0] * len(all_evaluation_types)
    nr_shared_best_perfomer_in_evaluation = [0] * len(all_evaluation_types)

    for j, evaluation_type in enumerate(all_evaluation_types):
      y_values = [0] * len(all_imputation_methods)

      for dataset in all_datasets:
        for miss_rate in all_miss_rates:
          value_gainv1 = float(get_filtered_values(df_summary, dataset=dataset, miss_rate=miss_rate, extra_amount=0, imputation_method="GAIN v1", evaluation=evaluation_type).values.ravel())
          value_gainv2 = float(get_filtered_values(df_summary, dataset=dataset, miss_rate=miss_rate, extra_amount=0, imputation_method="GAIN v2", evaluation=evaluation_type).values.ravel())

          if value_gainv1 == 0:
            continue
          else:
            nr_datasets_in_evaluation[j] += 1

          if args.evaluation_type == "Prediction" and dataset != "news": # Max is considered the best
            if value_gainv1 > value_gainv2:
               idx = 1
            elif value_gainv1 < value_gainv2:
               idx = 2
            else: 
               idx = 3
          elif args.evaluation_type == "Imputation" or (dataset == "news" and args.evaluation_type == "Prediction"): # Min is considered the best
            if value_gainv1 < value_gainv2:
               idx = 1
            elif value_gainv1 > value_gainv2:
               idx = 2
            else: 
               idx = 3
          
          if idx == 3:
             nr_shared_best_perfomer_in_evaluation[j] += 1
          else:
            y_values[(idx-1)] += 1
      
      all_y_values.append(y_values)

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(all_evaluation_types), figsize=(11, 3.5), sharex='col') 
   
    for i, evaluation_type in enumerate(all_evaluation_types):
      #x_ticks = np.arange(len(all_imputation_methods))
      bar_width = 0.25
      bar_positions = [0.4, 1]

      ax = axes[i]
      ax.bar(bar_positions, all_y_values[i], width=bar_width, color=colors)
      
      ax.set_xticks(bar_positions)
      ax.set_xticklabels(all_imputation_methods, rotation=0, ha='center', fontsize=9)
      ax.set_title(evaluation_labels[i])
      ax.text(0.5, -0.45, "# of datasets: " + str(nr_datasets_in_evaluation[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.text(0.5, -0.55, "# of shared best perfomer: " + str(nr_shared_best_perfomer_in_evaluation[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

      # Add horizontal lines for every integer on the y-axis
      for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
        ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)

    # Set titles
    #fig.suptitle('Number of best performer per evaluation metric for all baseline methods', y=0.98, fontsize=12)
    fig.text(0.03, 0.5, '# of best perfomer', va='center', rotation='vertical', fontsize=10) 

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.show()

def plotCTGANImpactNoBestResult(args, df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)
    all_imputation_methods = ['Median/mode', 'MICE', 'kNN', 'MissForest', 'GAIN v1', 'GAIN v2']
    all_datasets, all_miss_rates = args.all_datasets, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    bar_width = 0.4
    handles_list = []
    labels_list = []
    all_results = []
    no_results_not_stat_sign = [0] * len(all_imputation_methods)
    no_results_stat_sign = [0] * len(all_imputation_methods)

    for j, imputation_method in enumerate(all_imputation_methods):
        no_datasets = 0
        if imputation_method == "GAIN v1" or imputation_method == "GAIN v2":
            CTGAN_not_better_results = [0, 0, 0, 0]
            CTGAN_better_results = [0, 0, 0, 0]
        else: 
            CTGAN_not_better_results = [0, 0]
            CTGAN_better_results = [0, 0]

        for dataset in all_datasets:
            for miss_rate in all_miss_rates:
                values = [float(x) for x in get_filtered_values(df_summary, dataset=dataset, miss_rate=miss_rate, evaluation=evaluation, imputation_method=imputation_method).values.ravel()]
                if values[1] == 0: # data set was not added additional data
                    continue
                else:
                    no_datasets +=1

                if imputation_method == "GAIN v1" or imputation_method == "GAIN v2":
                    df_confidence = df_confidence_interval_gainv1 if imputation_method == "GAIN v1" else (df_confidence_interval_gainv2)

                    lower_0, upper_0 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "0 % CTGAN")
                    lower_50, upper_50 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "50 % CTGAN")
                    lower_100, upper_100 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "100 % CTGAN")
                    lower_200, upper_200 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "200 % CTGAN")
                    lower_500, upper_500 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "500 % CTGAN")
                    
                    indice_50 = find_best_value_stat_sign(lower_0, upper_0, lower_50, upper_50, evaluation)
                    indice_100 = find_best_value_stat_sign(lower_0, upper_0, lower_100, upper_100, evaluation)
                    indice_200 = find_best_value_stat_sign(lower_0, upper_0, lower_200, upper_200, evaluation)
                    indice_500 = find_best_value_stat_sign(lower_0, upper_0, lower_500, upper_500, evaluation)

                    indices = [indice_50, indice_100, indice_200, indice_500]

                    for k, indice in enumerate(indices):
                        if indice == 1: # 0 is better
                            CTGAN_not_better_results[k] += 1
                            no_results_stat_sign[j] += 1
                        elif indice == 2: #CTGAN is better
                            CTGAN_better_results[k] += 1
                            no_results_stat_sign[j] += 1
                        else:   
                            no_results_not_stat_sign[j] += 1
                elif imputation_method == "MissForest":
                    df_confidence = df_confidence_interval_missforest

                    lower_0, upper_0 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "0 % CTGAN")
                    lower_50, upper_50 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "50 % CTGAN")
                    lower_100, upper_100 = get_values_and_bounds(df_confidence, dataset, miss_rate, evaluation, "100 % CTGAN")
                    
                    indice_50 = find_best_value_stat_sign(lower_0, upper_0, lower_50, upper_50, evaluation)
                    indice_100 = find_best_value_stat_sign(lower_0, upper_0, lower_100, upper_100, evaluation)

                    indices = [indice_50, indice_100]

                    for k, indice in enumerate(indices):
                        if indice == 1: # 0 is better
                            CTGAN_not_better_results[k] += 1
                            no_results_stat_sign[j] += 1
                        elif indice == 2: #CTGAN is better
                            CTGAN_better_results[k] += 1
                            no_results_stat_sign[j] += 1
                        else:   
                            no_results_not_stat_sign[j] += 1
                else:
                    no_results_stat_sign[j] += 1

                    best_indice_50 = find_best_value(values[0], values[1], evaluation)
                    best_indice_100 = find_best_value(values[0], values[2], evaluation)
                    best_indices = [best_indice_50, best_indice_100]
            
                    for i, best_index in enumerate(best_indices):
                        if best_index == 2: # CTGAN is better
                            CTGAN_better_results[i] += 1
                        elif best_index == 1: # CTGAN is not better
                            CTGAN_not_better_results[i] += 1

        all_results.append(CTGAN_not_better_results)
        all_results.append(CTGAN_better_results)

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(all_imputation_methods), figsize=(11, 3.5), sharex='col')

    for i, imputation_method in enumerate(all_imputation_methods):
      dataset_values = all_results[i*2:(i+1)*2]
      x_ticks = np.arange(len(dataset_values[0]))
      xtick_pos = x_ticks + bar_width/2
      bars = []
      ax = axes[i]
      label_alternatives = ["CTGAN worsens result", "CTGAN improves result"]

      for j, label_alternative in enumerate(label_alternatives):
          dataset_values[j] = [float(x) for x in dataset_values[j]]
          bar = ax.bar(x_ticks+j*bar_width, dataset_values[j], width=bar_width, label=label_alternative, color=colors[j])
          bars.append(bar)
      
      ax.set_title(imputation_method)
      ax.set_xticks(xtick_pos)
      ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

      if imputation_method == "GAIN v1" or imputation_method == "GAIN v2":
        ax.set_xticklabels(["CTGAN 50%","CTGAN 100%", "CTGAN 200%", "CTGAN 500%"], rotation=35, ha='right', fontsize=8)
      else:
        ax.set_xticklabels(["CTGAN 50%","CTGAN 100%"], rotation=35, ha='right', fontsize=8)

      # Collect handles and labels for this subplot
      handles, labels = collect_handles_and_labels(bars, label_alternatives)
      handles_list.append(handles)
      labels_list.append(labels)

      # Add horizontal lines for every integer on the y-axis
      for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
        ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)
  
    # Set titles
    #fig.suptitle('CTGAN Impact', y=0.98, fontsize=12)
    #fig.text(0.5, 0.90, "Total number of datasets in comparison: " + str(no_datasets), fontsize=8, ha='center', va='bottom')
    fig.text(0.03, 0.5, '# of best perfomer in terms of ' + evaluation, va='center', rotation='vertical', fontsize=10) 
    fig.legend(handles_list[0], labels_list[0], loc='upper center', ncol=len(label_alternatives), bbox_to_anchor=(0.5, 0.11))
    
    print(no_results_not_stat_sign)
    print(no_results_stat_sign)

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.3, left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.show()


def plotBarChartNoBestResultBaselineMethodsStatSign(args, df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2):
    if args.evaluation_type == "Prediction":
       all_imputation_methods = args.all_imputation_methods
       all_evaluation_types = args.all_prediction_evaluations
    else:
       all_imputation_methods = ['Median/mode', 'MICE', 'kNN', 'MissForest', 'GAIN v1', 'GAIN v2']
       all_evaluation_types = args.all_imputation_evaluations
    
    all_y_values = []
    no_results_not_stat_sign = [0] * len(all_evaluation_types)
    no_results_stat_sign = [0] * len(all_evaluation_types)
    bar_width = 0.6
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    for j, evaluation in enumerate(all_evaluation_types):
        y_values = [0] * len(all_imputation_methods)
        # Find best value for every dataset
        for data_set in args.all_datasets:
            for miss_rate in args.all_miss_rates:

                # Check every confidence interval to see if the value is within that range
                mf_lower, mf_upper = get_values_and_bounds(df_confidence_interval_missforest, data_set, miss_rate, evaluation, "0 % CTGAN")
                if mf_lower == None and mf_upper == None:
                   continue
                
                gainv1_lower, gainv1_upper = get_values_and_bounds(df_confidence_interval_gainv1, data_set, miss_rate, evaluation, "0 % CTGAN")
                gainv2_lower, gainv2_upper = get_values_and_bounds(df_confidence_interval_gainv2, data_set, miss_rate, evaluation, "0 % CTGAN")
                                
                value_medianmode = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='Median/mode', evaluation=evaluation).values.ravel()[0])
                value_MICE = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='MICE', evaluation=evaluation).values.ravel()[0])
                value_kNN = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='kNN', evaluation=evaluation).values.ravel()[0])
                
                if args.evaluation_type == "Prediction":
                    value_listwise = float(get_filtered_values(df_summary, dataset=data_set, miss_rate=miss_rate, extra_amount=0, imputation_method='List-wise deletion', evaluation=evaluation).values.ravel()[0])
                    if value_listwise == 0:
                        if args.evaluation_type == "Prediction" and data_set != "news":
                            value_listwise = -np.inf
                        else:
                            value_listwise = np.inf

                    all_upper = [value_medianmode, value_MICE, value_kNN, mf_upper, gainv1_upper, gainv2_upper, value_listwise]
                    all_lower = [value_medianmode, value_MICE, value_kNN, mf_lower, gainv1_lower, gainv2_lower, value_listwise]
                elif args.evaluation_type == "Imputation":
                    all_upper = [value_medianmode, value_MICE, value_kNN, mf_upper, gainv1_upper, gainv2_upper]
                    all_lower = [value_medianmode, value_MICE, value_kNN, mf_lower, gainv1_lower, gainv2_lower]

                if args.evaluation_type == "Prediction" and data_set != "news": # Max is considered the best
                    highest_value = np.max(all_upper)
                    highest_index = np.argmax(all_upper)
                    lowest_value = all_lower[highest_index]
                    all_upper[highest_index] = -np.inf

                    if all(lowest_value >= val for val in all_upper):
                        # It is statistically significant
                        y_values[highest_index] += 1
                        no_results_stat_sign[j] += 1
                    else:
                        if highest_index in [4, 5]:
                            second_highest_index = np.argmax(all_upper)

                            if second_highest_index in [4, 5] and second_highest_index != highest_index:
                                second_lowest_value = all_lower[second_highest_index]
                                all_upper[second_highest_index] = -np.inf

                                if all(second_lowest_value >= val for val in all_upper):
                                    print("GAIN shares")
                        # It is not statistically significant
                        no_results_not_stat_sign[j] += 1
                    
                elif args.evaluation_type == "Imputation" or (data_set == "news" and args.evaluation_type == "Prediction"): # Min is considered the best
                    lowest_value = np.min(all_lower)
                    lowest_index = np.argmin(all_lower)
                    highest_value = all_upper[lowest_index]
                    all_lower[lowest_index] = np.inf

                    if all(highest_value <= val for val in all_lower):
                        # It is statistically significant
                        y_values[lowest_index] += 1
                        no_results_stat_sign[j] += 1
                    else:
                        if lowest_index in [4, 5]:
                            second_lowest_index = np.argmin(all_lower)

                            if second_lowest_index in [4, 5] and second_lowest_index != lowest_index:
                                second_highest_value = all_upper[second_lowest_index]
                                all_lower[second_lowest_index] = np.inf

                                if all(second_highest_value <= val for val in all_lower):
                                    print("GAIN shares")
                        # It is not statistically significant
                        no_results_not_stat_sign[j] += 1

        all_y_values.append(y_values)

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(all_evaluation_types), figsize=(11, 3.5), sharex='col') 
   
    for i, evaluation in enumerate(all_evaluation_types):
      x_ticks = np.arange(len(all_imputation_methods))
      xtick_pos = x_ticks + bar_width
      ax = axes[i]

      ax.bar(x_ticks+bar_width, all_y_values[i], width=bar_width, color=colors)

      ax.set_xticks(xtick_pos)
      ax.set_xticklabels(all_imputation_methods, rotation=30, ha='right', fontsize=8)
      ax.set_title(evaluation)
      ax.text(0.5, -0.45, "# of results sig. at 95% conf.: " + str(no_results_stat_sign[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.text(0.5, -0.55, "# of results not sig. at 95% conf.:  " + str(no_results_not_stat_sign[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

      # Add horizontal lines for every integer on the y-axis
      for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
        ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)

    # Set titles
    #fig.suptitle('Number of best performer per evaluation metric for all baseline methods', y=0.98, fontsize=12)
    fig.text(0.03, 0.5, '# of best perfomer', va='center', rotation='vertical', fontsize=10) 

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.show()

def plotBarChartNoBestResultAllMethodsStatSign(args, df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2):
    if args.evaluation_type == "Prediction":
       all_imputation_methods = args.all_imputation_methods
    else:
       all_imputation_methods = ['Median/mode', 'MICE', 'kNN', 'MissForest', 'GAIN v1', 'GAIN v2']
    
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)
    all_datasets, all_miss_rates = args.all_datasets, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    # Create x labels
    labels = []
    for method in all_imputation_methods:
      labels.append('{}'.format(method))
      if method != "List-wise deletion":
        labels.append('{} CTGAN 50%'.format(method))
        labels.append('{} CTGAN 100%'.format(method))
        if method == "GAIN v1" or method == "GAIN v2":
            labels.append('{} CTGAN 200%'.format(method))
            labels.append('{} CTGAN 500%'.format(method))

    y_values = [0] * len(labels)
    no_results_stat_sign = 0
    no_results_not_stat_sign = 0

    for data_set in all_datasets:
       for miss_rate in all_miss_rates:
            if miss_rate == 50 or (data_set == "news" and miss_rate == 30):
                continue
            
            all_lower, all_upper = get_all_upper_and_lower(df_summary, df_confidence_interval_missforest, df_confidence_interval_gainv1, df_confidence_interval_gainv2, data_set, miss_rate, evaluation)
            
            if all_lower == None and all_upper == None:
                continue
            
            if evaluation == "AUROC" or evaluation == "Accuracy": # Max is considered the best
                highest_value = np.max(all_upper)
                highest_index = np.argmax(all_upper)
                lowest_value = all_lower[highest_index]
                all_upper[highest_index] = -np.inf

                if all(lowest_value >= val for val in all_upper):
                    # It is statistically significant
                    y_values[highest_index] += 1
                    no_results_stat_sign += 1
                else:
                    if highest_index in [12,13,14,15,16,17,18,19,20,21]:
                        second_highest_index = np.argmax(all_upper)

                        if second_highest_index in [12,13,14,15,16,17,18,19,20,21] and second_highest_index != highest_index:
                            second_lowest_value = all_lower[second_highest_index]
                            all_upper[second_highest_index] = -np.inf

                            if all(second_lowest_value >= val for val in all_upper):
                                print("GAIN shares")
                    # It is not statistically significant
                    no_results_not_stat_sign += 1   
            else:  # Min is considered the best
                lowest_value = np.min(all_lower)
                lowest_index = np.argmin(all_lower)
                highest_value = all_upper[lowest_index]
                all_lower[lowest_index] = np.inf

                if all(highest_value <= val for val in all_lower):
                    # It is statistically significant
                    y_values[lowest_index] += 1
                    no_results_stat_sign += 1
                else:
                    if lowest_index in [12,13,14,15,16,17,18,19,20,21]:
                        second_lowest_index = np.argmin(all_lower)

                        if second_lowest_index in [12,13,14,15,16,17,18,19,20,21] and second_lowest_index != lowest_index:
                            second_highest_value = all_upper[second_lowest_index]
                            all_lower[second_lowest_index] = np.inf

                            if all(second_highest_value <= val for val in all_lower):
                                print("GAIN shares")
                    # It is not statistically significant
                    no_results_not_stat_sign += 1

    # Create figure and set titles
    fig, ax = plt.subplots(figsize=(8,4))
    #fig.suptitle('Number of best perfomer in terms of '+ evaluation + ' for all methods', y=0.98, fontsize=12)
    fig.text(0.5, 0.88, "# of results sig. at 95% conf.: " + str(no_results_stat_sign), fontsize=8, ha='center', va='bottom')
    fig.text(0.5, 0.85, "# of results not sig. at 95% conf.: " + str(no_results_not_stat_sign), fontsize=8, ha='center', va='bottom')

    fig.text(0.03, 0.5, '# of best perfomer in terms of ' + evaluation, va='center', rotation='vertical', fontsize=10) 
    fig.subplots_adjust(hspace=0.6, wspace=0.3, left=0.1, right=0.9, top=0.8, bottom=0.3)

    ax.bar(labels, y_values, color=colors)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=35, ha='right')
    #plt.subplots_adjust(bottom=0.2)
    plt.xticks(fontsize=8)

    # Add horizontal lines for every integer on the y-axis
    for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
      ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--all_datasets',
      default=["mushroom", "letter", "bank", "credit", "news"],
      type=str)
    parser.add_argument(
      '--all_miss_rates',
      default=[10, 30, 50],
      type=str)
    parser.add_argument(
      '--all_imputation_methods',
      default=['Median/mode', 'MICE', 'kNN', 'MissForest', 'GAIN v1', 'GAIN v2', 'List-wise deletion'],
      type=str)
    parser.add_argument(
      '--imputation_evaluation',
      choices = ['mRMSE', 'RMSE num', 'RMSE cat', 'PFC (%)', 'Execution time (seconds)'],
      default='mRMSE',
      type=str)
    parser.add_argument(
      '--prediction_evaluation',
      choices = ['Accuracy', 'AUROC', 'MSE'],
      default='AUROC',
      type=str)
    parser.add_argument(
      '--evaluation_type',
      choices = ['Imputation', 'Prediction'],
      default='I',
      type=str)
    parser.add_argument(
      '--all_imputation_evaluations',
      default = ['mRMSE', 'RMSE num', 'RMSE cat', 'PFC (%)', 'Execution time (seconds)'],
      type=str)
    parser.add_argument(
      '--all_prediction_evaluations',
      default = ['Accuracy', 'AUROC', 'MSE'],
      type=str)
  
    args = parser.parse_args()

    # Set parameters
    args.imputation_evaluation = 'mRMSE'
    args.prediction_evaluation = 'AUROC'
    args.evaluation_type = 'Prediction'

    main(args)