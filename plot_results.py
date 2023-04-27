import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import brewer2mpl
import pandas as pd
import numpy as np
import math

from utils import find_miss_rates, collect_handles_and_labels, \
  is_vector_all_zeros, get_filtered_values, find_no_training_samples, get_filtered_values_separateCsv, \
  is_matrix_all_zeros, find_evaluation_type, find_best_value, get_filtered_values_df

def plotCTGANImpact(args, all_df_imputation, all_df_prediction):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)
    if args.evaluation_type == "Prediction":
       all_df = all_df_prediction
    else:
       all_df = all_df_imputation
    
    all_datasets, imputation_method, ctgan_options = args.all_datasets, args.imputation_method, args.all_ctgan_options
    values_list = []
    current_datasets = []
    handles_list = []
    labels_list = []
    bar_width = 0.18
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    for dataset in all_datasets:
      values_ctgan50 = get_filtered_values_df(all_df[1], dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()
      values_ctgan50 = values_ctgan50[values_ctgan50 != 0]
      if not is_vector_all_zeros(values_ctgan50):
          current_datasets.append(dataset)
          values_ctgan100 = get_filtered_values_df(all_df[2], dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()
          values = get_filtered_values_df(all_df[0], dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()[:len(values_ctgan50)]
          
          values_list.extend([values, values_ctgan50, values_ctgan100])
          values_ctgan200 = get_filtered_values_df(all_df[3], dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()
          values_ctgan500 = get_filtered_values_df(all_df[4], dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()
          if not is_vector_all_zeros(values_ctgan200):
            values_list.extend([values_ctgan200, values_ctgan500])
          else:
            ctgan_options = ctgan_options[0:3]

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(current_datasets), figsize=(11, 3.5), sharex='col') 

    for i, dataset in enumerate(current_datasets):
      dataset_values = values_list[i*len(ctgan_options):(i+1)*len(ctgan_options)]
      x_ticks = np.arange(len(dataset_values[0]))
      xtick_pos = x_ticks + 2*bar_width
      bars = []
     
      if len(current_datasets)>1:
        ax = axes[i]
      else: 
        ax = axes

      miss_rates = find_miss_rates(dataset_values[0])
      miss_rates_with_percent = [str(value) + '%' for value in miss_rates]

      for j, ctgan_option in enumerate(ctgan_options):
          dataset_values[j] = [float(x) for x in dataset_values[j]]
          bar = ax.bar(x_ticks+j*bar_width, dataset_values[j], width=bar_width, label=ctgan_option, color=colors[j])
          bars.append(bar)

      ax.set_xticks(xtick_pos)
      ax.set_xticklabels(miss_rates_with_percent)
      ax.set_title(dataset)
      y_min = np.nanmin(dataset_values)
      y_max = np.nanmax(dataset_values) 
      y_range = y_max - y_min
      ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)  # Add 5% margins on both sides

      if evaluation == 'Execution time (seconds)':
        ax.set_yscale('log')

      # Collect handles and labels for this subplot
      handles, labels = collect_handles_and_labels(bars, ctgan_options)
      handles_list.append(handles)
      labels_list.append(labels)

    # Set titles
    fig.suptitle(str(imputation_method) + ' for all datasets with different additional CTGAN data', y=0.98, fontsize=12)
    fig.text(0.03, 0.5, evaluation, va='center', rotation='vertical', fontsize=10) 
    fig.legend(handles_list[0], labels_list[0], loc='upper center', ncol=len(ctgan_options), bbox_to_anchor=(0.5, 0.15))

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.25)
    plt.show()

def plotCTGANImpactNoBestResult(args, df):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)
    all_datasets, all_imputation_methods, all_miss_rates = args.all_datasets, args.all_imputation_methods, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    bar_width = 0.4
    handles_list = []
    labels_list = []
    all_results = []

    for imputation_method in all_imputation_methods:
      no_datasets = 0
      if imputation_method == "GAIN v1" or imputation_method == "GAIN v2":
        CTGAN_not_better_results = [0, 0, 0, 0]
        CTGAN_better_results = [0, 0, 0, 0]
      else: 
        CTGAN_not_better_results = [0, 0]
        CTGAN_better_results = [0, 0]

      for dataset in all_datasets:
        for miss_rate in all_miss_rates:
          values = [float(x) for x in get_filtered_values_df(df, dataset=dataset, miss_rate=miss_rate, evaluation=evaluation, imputation_method=imputation_method).values.ravel()]
          if values[1] == 0: # data set was not added additional data
             continue
          else:
             no_datasets +=1

          if values[4] == 0: 
            best_indices = [find_best_value(values[0], values[i], evaluation) for i in [1,2]]
          else: # 200% and 500% also is part
            best_indices = [find_best_value(values[0], values[i], evaluation) for i in [1,2,3,4]]
          
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
      label_alternatives = ["Not improved compared to no additional data", "Improved compared to no additional data"]

      for j, label_alternative in enumerate(label_alternatives):
          dataset_values[j] = [float(x) for x in dataset_values[j]]
          bar = ax.bar(x_ticks+j*bar_width, dataset_values[j], width=bar_width, label=label_alternative, color=colors[j+3])
          bars.append(bar)
      
      ax.set_title(imputation_method)
      ax.set_xticks(xtick_pos)
      if len(dataset_values[0]) == 2:
        ax.set_xticklabels(["CTGAN 50%","CTGAN 100%"], rotation=35, ha='right', fontsize=8)
      else:
        ax.set_xticklabels(["CTGAN 50%","CTGAN 100%", "CTGAN 200%", "CTGAN 500%"], rotation=35, ha='right', fontsize=8)
      

      if evaluation == 'Execution time (seconds)':
        ax.set_yscale('log')

      # Collect handles and labels for this subplot
      handles, labels = collect_handles_and_labels(bars, label_alternatives)
      handles_list.append(handles)
      labels_list.append(labels)

      # Add horizontal lines for every integer on the y-axis
      for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
        ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)
  
    # Set titles
    fig.suptitle('CTGAN Impact', y=0.98, fontsize=12)
    #fig.text(0.5, 0.90, "Total number of datasets in comparison: " + str(no_datasets), fontsize=8, ha='center', va='bottom')
    fig.text(0.03, 0.5, '# of best perfomer in terms of ' + evaluation, va='center', rotation='vertical', fontsize=10) 
    fig.legend(handles_list[0], labels_list[0], loc='upper center', ncol=len(label_alternatives), bbox_to_anchor=(0.5, 0.15))

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.3, left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.show()

def plotBarChartNoBestResultBaselineMethods(args,  df):
    if args.evaluation_type == "Prediction":
       all_evaluation_types = args.all_prediction_evaluations
    else:
       all_evaluation_types = args.all_imputation_evaluations
    all_datasets, all_imputation_methods, all_miss_rates = args.all_datasets, args.all_imputation_methods, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    bar_width = 0.6
    all_y_values = []
    nr_datasets_in_evaluation = [0] * len(all_evaluation_types)
    nr_shared_best_perfomer_in_evaluation = [0] * len(all_evaluation_types)

    for j, evaluation_type in enumerate(all_evaluation_types):
      y_values = [0] * len(all_imputation_methods)

      for dataset in all_datasets:
        for miss_rate in all_miss_rates:
          values = get_filtered_values(df, dataset=dataset, miss_rate=miss_rate, extra_amount=0, evaluation=evaluation_type).values.ravel()
          
          if is_vector_all_zeros(values):
            continue
          else:
            nr_datasets_in_evaluation[j] += 1

          if args.evaluation_type == "Prediction" and dataset != "news": # Max is considered the best
            idx = np.where(values == np.max(values))[0]
          elif args.evaluation_type == "Imputation" or (dataset == "news" and args.evaluation_type == "Prediction"): # Min is considered the best
            idx = np.where(values == np.min(values))[0]
          
          if len(idx) > 1:
             nr_shared_best_perfomer_in_evaluation[j] += 1
          
          for i in idx:
            y_values[i] += 1
      
      all_y_values.append(y_values)

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(all_evaluation_types), figsize=(11, 3.5), sharex='col') 

    for i, evaluation_type in enumerate(all_evaluation_types):
      x_ticks = np.arange(len(all_imputation_methods))
      xtick_pos = x_ticks + bar_width
      ax = axes[i]

      ax.bar(x_ticks+bar_width, all_y_values[i], width=bar_width, color=colors)

      ax.set_xticks(xtick_pos)
      ax.set_xticklabels(all_imputation_methods, rotation=45, ha='right', fontsize=8)
      ax.set_title(evaluation_type)
      ax.text(0.5, -0.45, "# of datasets: " + str(nr_datasets_in_evaluation[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.text(0.5, -0.55, "# of shared best perfomer: " + str(nr_shared_best_perfomer_in_evaluation[i]), fontsize=7, ha='center', transform=ax.transAxes)
      ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
      ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

      # Add horizontal lines for every integer on the y-axis
      for i in range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1])+1):
        ax.axhline(y=i, color='grey', linestyle='--', linewidth=0.2)

    # Set titles
    fig.suptitle('Number of best performer per evaluation metric for all baseline methods', y=0.98, fontsize=12)
    fig.text(0.03, 0.5, '# of best perfomer', va='center', rotation='vertical', fontsize=10) 

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.3)
    plt.show()
    

def plotBarChartNoBestResultAllMethods(args, df):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)
    all_datasets, all_imputation_methods, all_miss_rates = args.all_datasets, args.all_imputation_methods, args.all_miss_rates
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    # Create x labels
    labels = []
    for method in all_imputation_methods:
      labels.append('{}'.format(method))
      labels.append('{} CTGAN 50%'.format(method))
      labels.append('{} CTGAN 100%'.format(method))

    y_values = [0] * len(labels)
    no_datasets = 0
    no_shared_first_places = 0

    for dataset in all_datasets:
       for miss_rate in all_miss_rates:
          df_filtered = get_filtered_values(df, dataset=dataset, miss_rate=miss_rate, evaluation=evaluation)
          matrix = df_filtered.values.astype(float)

          if is_matrix_all_zeros(matrix) or np.any(np.all(matrix == 0, axis=1)):
             continue
          
          no_datasets += 1

          if args.evaluation_type == "Prediction" and dataset != "news": # Max is considered the best
            nonzero_matrix = np.where(matrix != 0, matrix, -np.inf)  # Replace zeros with infinity
            value = np.nanmax(nonzero_matrix)
          elif args.evaluation_type == "Imputation" or (dataset == "news" and args.evaluation_type == "Prediction"): # Min is considered the best
            nonzero_matrix = np.where(matrix != 0, matrix, np.inf)  # Replace zeros with negative infinity
            value = np.nanmin(nonzero_matrix)
          
          row_indexes, col_indexes = np.where(matrix == value)

          if len(row_indexes) > 1:
             no_shared_first_places += 1
          
          for i, row_index in enumerate(row_indexes):
            if row_index == 0:
              best_extra_amount = ""
            elif row_index == 1:
              best_extra_amount = " CTGAN 50%"
            elif row_index == 2:
              best_extra_amount = " CTGAN 100%"  
            
            best_imputation_method = df_filtered.columns.get_level_values(0)[col_indexes[i]]
            tot_best_method = '{}{}'.format(best_imputation_method, best_extra_amount)

            # loop through the labels and check if it matches
            for j, label in enumerate(labels):
              if tot_best_method == label:
                y_values[j] += 1
                break

    # Create figure and set titles
    fig, ax = plt.subplots(figsize=(8,4))
    fig.suptitle('Number of best perfomer in terms of '+ evaluation + ' for all methods', y=0.98, fontsize=12)
    fig.text(0.5, 0.88, "Total number of datasets in comparison: " + str(no_datasets), fontsize=8, ha='center', va='bottom')
    fig.text(0.5, 0.85, "Total number of shared best perfomer: " + str(no_shared_first_places), fontsize=8, ha='center', va='bottom')

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

def plotCTGANImpact_OLD(args, df):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)
    all_datasets, imputation_method, ctgan_options = args.all_datasets, args.imputation_method, args.all_ctgan_options
    values_list = []
    current_datasets = []
    handles_list = []
    labels_list = []
    bar_width = 0.25
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    for dataset in all_datasets:
      values_ctgan50 = get_filtered_values(df, extra_amount=50, dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()
      values_ctgan50 = values_ctgan50[values_ctgan50 != 0]
      if not is_vector_all_zeros(values_ctgan50):
          current_datasets.append(dataset)
          values_ctgan100 = get_filtered_values(df, extra_amount=100, dataset=dataset, miss_rate=None, imputation_method=imputation_method, evaluation=evaluation).values.ravel()[:len(values_ctgan50)]
          values = get_filtered_values(df, extra_amount=0, dataset=dataset, miss_rate=None,  imputation_method=imputation_method, evaluation=evaluation).values.ravel()[:len(values_ctgan50)]
          values_list.extend([values, values_ctgan50, values_ctgan100])

    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(current_datasets), figsize=(11, 3.5), sharex='col') 

    for i, dataset in enumerate(current_datasets):
      dataset_values = values_list[i*len(ctgan_options):(i+1)*len(ctgan_options)]
      x_ticks = np.arange(len(dataset_values[0]))
      xtick_pos = x_ticks + bar_width
      bars = []
     
      if len(current_datasets)>1:
        ax = axes[i]
      else: 
        ax = axes

      miss_rates = find_miss_rates(dataset_values[0])
      miss_rates_with_percent = [str(value) + '%' for value in miss_rates]

      for j, ctgan_option in enumerate(ctgan_options):
          dataset_values[j] = [float(x) for x in dataset_values[j]]
          bar = ax.bar(x_ticks+j*bar_width, dataset_values[j], width=bar_width, label=ctgan_option, color=colors[j])
          bars.append(bar)

      ax.set_xticks(xtick_pos)
      ax.set_xticklabels(miss_rates_with_percent)
      ax.set_title(dataset)
      y_min = np.nanmin(dataset_values)
      y_max = np.nanmax(dataset_values) 
      y_range = y_max - y_min
      ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)  # Add 5% margins on both sides

      if evaluation == 'Execution time (seconds)':
        ax.set_yscale('log')

      # Collect handles and labels for this subplot
      handles, labels = collect_handles_and_labels(bars, ctgan_options)
      handles_list.append(handles)
      labels_list.append(labels)

    # Set titles
    fig.suptitle(str(imputation_method) + ' for all datasets with different additional CTGAN data', y=0.98, fontsize=12)
    fig.text(0.03, 0.5, evaluation, va='center', rotation='vertical', fontsize=10) 
    fig.legend(handles_list[0], labels_list[0], loc='upper center', ncol=len(ctgan_options), bbox_to_anchor=(0.5, 0.15))

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.25)
    plt.show()

def plotBarChartAllDatasetsAllMissingness(args, df, extra_amount):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)

    all_datasets, imputation_methods = args.all_datasets, args.all_imputation_methods
    values_list = []
    current_datasets = []
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    handles_list = []
    labels_list = []
    bar_width = 0.17

    for dataset in all_datasets:
      for i, imputation_method in enumerate(imputation_methods):
        dataset_values = get_filtered_values_separateCsv(df, dataset=dataset, imputation_method=imputation_method, evaluation=evaluation)
        if is_vector_all_zeros(dataset_values) == False:
            if dataset not in current_datasets:
                current_datasets.append(dataset)
            values_list.append(dataset_values)
         
    ## Create a figure with  subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(current_datasets), figsize=(10, 3), sharex='col')

    for i, dataset in enumerate(current_datasets):
      dataset_values = values_list[i*len(imputation_methods):(i+1)*len(imputation_methods)]
      x_ticks = np.arange(len(dataset_values[0]))
      xtick_pos = x_ticks + bar_width*2
      ax = axes[i]
      bars = []
      miss_rates = find_miss_rates(dataset_values[0])
      miss_rates_with_percent = [str(value) + '%' for value in miss_rates]

      ax.set_xticks(xtick_pos)
      ax.set_xticklabels(miss_rates_with_percent)
      ax.set_title(dataset)
      y_min = np.min(dataset_values)
      y_max = np.max(dataset_values) 
      y_range = y_max - y_min
      ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)  # Add 5% margins on both sides

      for j, imputation_method in enumerate(imputation_methods):
          bar = ax.bar(x_ticks+j*bar_width, dataset_values[j], width=bar_width, label=imputation_method, color=colors[j])
          bars.append(bar)

      if evaluation == 'Execution time (seconds)':
        ax.set_yscale('log')

      # Collect handles and labels for this subplot
      handles, labels = collect_handles_and_labels(bars, imputation_methods)
      handles_list.append(handles)
      labels_list.append(labels)

    # Set title for the whole figure
    label = "no additional CTGAN data"
    if extra_amount == 50:
     label = "50% additional CTGAN data"
    elif extra_amount == 100:
      label = "100% additional CTGAN data"

    # Set titles
    fig.suptitle(str(evaluation) +' for '+ label, y=0.98, fontsize=12)
    fig.text(0.03, 0.5, evaluation, va='center', rotation='vertical', fontsize=10) 
    fig.legend(handles_list[0], labels_list[0], loc='upper center', ncol=len(imputation_methods), bbox_to_anchor=(0.5, 0.15))

    # Show the plot
    fig.subplots_adjust(hspace=0.6, wspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.25)
    plt.show()
  
'''def plotBarChartAllDatasetsOneMissingness(args, df, extra_amount):
    evaluation = find_evaluation_type(args.evaluation_type, args.imputation_evaluation, args.prediction_evaluation)

    all_datasets, miss_rate, imputation_methods = args.all_datasets, args.miss_rate, args.all_imputation_methods
    values_list = []
    current_datasets = []
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    for dataset in all_datasets:
        dataset_values = get_filtered_values_separateCsv(df, dataset=dataset, miss_rate=None, imputation_method=None, evaluation=evaluation)
        if is_vector_all_zeros(dataset_values) == False:
            if dataset not in current_datasets:
                current_datasets.append(dataset)
            values_list.append(dataset_values)

    ## Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(current_datasets), figsize=(11, 3.5), sharex='col')
    fig.text(0.005, 0.5, evaluation, va='center', rotation='vertical', fontsize=10)  

    # Plot bar charts for each dataset
    handles_list = []
    labels_list = []

    for i, data in enumerate(current_datasets):
        ax = axes[i]
        bars = ax.bar(imputation_methods, values_list[i], color=colors)

        if evaluation == 'Execution time (seconds)':
          ax.set_yscale('log')

        ax.set_title(current_datasets[i])
        ax.xaxis.set_tick_params(which='both', length=0)
        ax.set_xticklabels([])

        # Collect handles and labels for this subplot
        handles, labels = collect_handles_and_labels(bars, imputation_methods)
        handles_list.append(handles)
        labels_list.append(labels)

    # Set title for the whole figure
    label = "no additional CTGAN data"
    if extra_amount == 50:
     label = "50% additional CTGAN data"
    elif extra_amount == 100:
      label = "100% additional CTGAN data"
    fig.suptitle(str(evaluation) +' with '+str(miss_rate)+ '% missingness and '+ label)

    # Add some margin to the top of the subplots and move the legend there
    fig.subplots_adjust(top=0.2)
    fig.legend(handles, labels, loc='upper center', ncol=len(imputation_methods), bbox_to_anchor=(0.5, 0.9))

    # Show the plot
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.2, left=0.1, top=0.5)
    fig.tight_layout(pad=2)
    plt.show()'''

'''
def plotNoTrainingSamplesImpact(args, df, df_ctgan50, df_ctgan100):
  all_miss_rates, all_datasets, all_imputation_methods, all_ctgan_options = args.all_miss_rates, args.all_datasets, args.all_imputation_methods, args.ctgan_options
  if args.evaluation_type == "Prediction":
    evaluation = args.prediction_evaluation
  elif args.evaluation_type == "Imputation":
    evaluation = args.imputation_evaluation

  all_x = []
  all_y = []

  for miss_rate in all_miss_rates:
      for imputation_method in all_imputation_methods:
        y_values = []
        x_values = []
        for dataset in all_datasets:
          for ctgan_option in all_ctgan_options:
            if ctgan_option == "No CTGAN":
              y_value = get_filtered_values(df, dataset=dataset, imputation_method=imputation_method, evaluation=evaluation, miss_rate=miss_rate)
            elif ctgan_option == "CTGAN 50%": 
              y_value = get_filtered_values(df_ctgan50, dataset=dataset, imputation_method=imputation_method, evaluation=evaluation, miss_rate=miss_rate)
            elif ctgan_option == "CTGAN 100%":
              y_value = get_filtered_values(df_ctgan100, dataset=dataset, imputation_method=imputation_method, evaluation=evaluation, miss_rate=miss_rate)
            
            if y_value:
              x_value = find_no_training_samples(ctgan_option, dataset)
              y_values.append(y_value[0])
              x_values.append(x_value)

        all_y.append(y_values)
        all_x.append(x_values)
   
  ## Create a figure with subplots
  fig, axes = plt.subplots(nrows=1, ncols=len(all_miss_rates), figsize=(11, 3.5), sharex='col')
  fig.text(0.005, 0.5, evaluation, va='center', rotation='vertical', fontsize=10)  

  # Plot bar charts for each dataset
  handles_list = []
  labels_list = []

  for i, miss_rate in enumerate(all_miss_rates):
    ax = axes[i]

    start_index = i * 5
    end_index = start_index + 5
    subvector_x = all_x[start_index:end_index]
    subvector_y = all_y[start_index:end_index]

    for j, imputation_method in enumerate(all_imputation_methods):
      subvector_x[j], subvector_y[j] = zip(*sorted(zip(subvector_x[j], subvector_y[j])))
      ax.plot(subvector_x[j], subvector_y[j], label=imputation_method)

    if evaluation == 'Execution time (seconds)':
      ax.set_yscale('log')

    ax.set_title(all_miss_rates[i])
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.set_xticklabels([])

    # Collect handles and labels for this subplot
    #handles, labels = collect_handles_and_labels(bars, imputation_methods)
    #handles_list.append(handles)
    #labels_list.append(labels)

    fig.suptitle(str(evaluation) +' with '+str(miss_rate)+ '% missingness and ')

    # Add some margin to the top of the subplots and move the legend there
    fig.subplots_adjust(top=0.2)
    #fig.legend(handles, labels, loc='upper center', ncol=len(all_miss_rates), bbox_to_anchor=(0.5, 0.9))

    # Show the plot
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.2, left=0.1, top=0.5)
    fig.tight_layout(pad=2)
    plt.show()'''