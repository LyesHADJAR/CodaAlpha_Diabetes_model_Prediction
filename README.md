# Diabetes Prediction Using Machine Learning Models

This project aims to predict the likelihood of diabetes in patients using various machine learning models. The data is preprocessed, and multiple models are trained and evaluated to determine their accuracy in predicting diabetes.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The objective of this project is to preprocess the dataset, apply different machine learning algorithms, and evaluate their performance in predicting diabetes. The models used in this project include RandomForest, LogisticRegression, SVM, KNeighbors, GradientBoosting, XGBoost, LightGBM, NaiveBayes, NeuralNetwork, and AdaBoost.

## Dataset
The dataset contains various features related to patients' health and demographics. It includes columns such as gender, location, and other health-related metrics. The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical features.

## Pre-requisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - lightgbm

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/diabetes-prediction.git
    cd src
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open the `code_with_data_preprocessing.ipynb` notebook.
3. Run the cells in the notebook to preprocess the data, train the models, and evaluate their performance.

## Models
The following models are created and evaluated in this project:
- RandomForestClassifier
- LogisticRegression
- SVC (Support Vector Classifier)
- KNeighborsClassifier
- GradientBoostingClassifier
- XGBoostClassifier
- LightGBMClassifier
- GaussianNB (Naive Bayes)
- MLPClassifier (Neural Network)
- AdaBoostClassifier

## Results
The accuracy of each model is calculated and displayed using a bar plot. The models are evaluated based on their accuracy scores, which are stored in a dictionary and then converted to a DataFrame for visualization.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
