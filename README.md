# Machine Learning Pipeline for Tourism Demand Prediction

## Project Overview

This project builds a machine learning pipeline to predict tourism demand across multiple countries using tourism and economic indicators.

The system includes data preprocessing, feature engineering, model training, experiment tracking, and automatic model selection.

The goal is to evaluate multiple machine learning models and identify the best-performing model for predicting inbound tourism arrivals.

## Machine Learning Models Used

The project compares several regression models:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Neural Network (PyTorch)

Model performance is evaluated using standard regression metrics.

## Features of the Project

The pipeline includes:

* Data preprocessing (missing values, duplicates, outlier removal)
* Feature engineering
* Multiple machine learning models
* Neural network implementation using PyTorch
* Model evaluation using RMSE, MAE, and R²
* Experiment tracking
* Automatic best model selection

## Project Structure

tourism-demand-ml
│
├── data
├── notebooks
│   └── exploratory_analysis.ipynb
│
├── results
│   └── model_results.csv
│
├── src
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── neural_network.py
│   ├── train_models.py
│   └── experiment_tracker.py
│
├── main.py
├── requirements.txt
└── README.md

## Evaluation Metrics

The models are evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

## Dataset

The dataset used in this project was compiled from multiple tourism statistics sources during academic research.

Due to aggregation from multiple sources, the dataset is not included in this repository.

The pipeline expects the dataset in the following location:

data/tourism_dataset.csv

## How to Run the Project

Install the required dependencies:

pip install -r requirements.txt

Run the main pipeline:

python main.py

## Output

The pipeline will:

* Train all models
* Evaluate their performance
* Save experiment results to:

results/model_results.csv

* Automatically identify the best-performing model.

## Author

Suhas Ramesh