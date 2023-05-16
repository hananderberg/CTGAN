from plot_results import plotBarChartAllDatasetsAllMissingness, \
plotCTGANImpact, plotBarChartNoBestResultAllMethods, plotCTGANImpact_OLD, plotCTGANImpactNoBestResult, plotBarChartNoBestResultBaselineMethods, plotTablePerDataset
from utils import get_filtered_values_df_confidence_interval, readDataConfidenceInterval, readDataSeparateCsv, readDataSeparateCsvBothImputationAndPrediction, readDataSummary
import argparse


def main(args):
    ## Load data different ways
    #df_confidence_interval = readDataConfidenceInterval("MissForest")
    df, df_ctgan50, df_ctgan100 = readDataSeparateCsv(args)
    df_summary = readDataSummary(args)
    imputation_df, imputation_df_ctgan50, imputation_df_ctgan100, imputation_df_ctgan200, imputation_df_ctgan500, \
    prediction_df, prediction_df_ctgan50, prediction_df_ctgan100, prediction_df_ctgan200, prediction_df_ctgan500 = readDataSeparateCsvBothImputationAndPrediction()
    all_df_imputation = [imputation_df, imputation_df_ctgan50, imputation_df_ctgan100, imputation_df_ctgan200, imputation_df_ctgan500]
    all_df_prediction = [prediction_df, prediction_df_ctgan50, prediction_df_ctgan100, prediction_df_ctgan200, prediction_df_ctgan500]

    ## Different plots
    #plotBarChartNoBestResultAllMethods(args, df_summary)
    plotBarChartNoBestResultBaselineMethods(args, df_summary)

    #plotTablePerDataset(args, df_summary) 

    #plotCTGANImpact_OLD(args, df_summary)

    #plotCTGANImpact(args, df_summary)

    #plotBarChartAllDatasetsAllMissingness(args, df, 0)
    #plotBarChartAllDatasetsAllMissingness(args, df_ctgan50, 50)
    #plotBarChartAllDatasetsAllMissingness(args, df_ctgan100, 100)

    #plotCTGANImpactNoBestResult(args, df_summary)

if __name__ == '__main__':   
    # Inputs for the main function
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
      '--all_ctgan_options',
      default=["Baseline", "CTGAN 50%", "CTGAN 100%", "CTGAN 200%", "CTGAN 500%"],
      type=str)
    parser.add_argument(
      '--all_imputation_methods',
      default=['Median/mode', 'MICE', 'kNN', 'MissForest', 'GAIN v1', 'GAIN v2', 'List-wise deletion'],
      type=str)
    parser.add_argument(
      '--miss_rate',
      choices=[10, 30, 50],
      default=10,
      type=str)
    parser.add_argument(
      '--imputation_method',
      choices=['Median/mode', 'MICE', 'kNN', 'MissForest', 'GAIN v1', 'GAIN v2', 'List-wise deletion'],
      default='MissForest',
      type=str)
    parser.add_argument(
      '--all_imputation_evaluations',
      default = ['mRMSE', 'RMSE num', 'RMSE cat', 'PFC (%)', 'Execution time (seconds)'],
      type=str)
    parser.add_argument(
      '--all_prediction_evaluations',
      default = ['Accuracy', 'AUROC', 'MSE'],
      type=str)
    parser.add_argument(
      '--imputation_evaluation',
      choices = ['mRMSE', 'RMSE num', 'RMSE cat', 'PFC (%)', 'Execution time (seconds)'],
      default='mRMSE',
      type=str)
    parser.add_argument(
      '--prediction_evaluation',
      choices = ['Accuracy', 'AUROC', 'MSE'],
      default='Accuracy',
      type=str)
    parser.add_argument(
      '--evaluation_type',
      choices = ['Imputation', 'Prediction'],
      default='Imputation',
      type=str)
    parser.add_argument(
      '--data_set',
      choices = ["mushroom", "letter", "bank", "credit", "news"],
      default='mushroom',
      type=str)
  
    args = parser.parse_args()

    # Set parameters
    args.imputation_evaluation = 'mRMSE'
    args.prediction_evaluation = 'AUROC'
    args.evaluation_type = 'Imputation'
    args.imputation_method = 'GAIN v2'
    args.data_set = "news"
    main(args)