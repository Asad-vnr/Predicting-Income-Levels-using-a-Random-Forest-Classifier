# Predicting-Income-Levels-using-a-Random-Forest-Classifier

Project Overview

This project is a comprehensive machine learning case study focused on predicting whether an individual's annual income exceeds $50,000. Using the "Adult Census Income" dataset, this project implements a Random Forest Classifier to perform binary classification.

The workflow covers all essential stages of a data science project, including data loading, extensive preprocessing, outlier management, model training, and in-depth evaluation, providing a robust template for tackling similar classification problems.

Dataset

The project utilizes the classic "Adult Census Income" dataset, which is a well-known benchmark for classification tasks.

    Source: Kaggle (originally from the UCI Machine Learning Repository)

    Link: https://www.kaggle.com/datasets/uciml/adult-census-income

    File Used: adult.csv

The dataset contains 15 columns of socio-economic data for individuals, such as age, education level, marital status, and hours worked per week. The goal is to predict the income column, which is a binary variable: <=50K or >50K.

Project Workflow

The project is structured with a clear, step-by-step methodology:

    Data Loading and Initial Cleaning:

        The adult.csv dataset is loaded into a pandas DataFrame.

        Non-standard missing values, represented as ' ?', are correctly identified during the loading process.

        Column names are immediately standardized by removing spaces and replacing dots (.) and hyphens (-) with underscores (_) for easier access.

    Exploratory Data Analysis (EDA):

        An initial investigation is performed to understand the dataset's structure, data types, and statistical properties.

        The distribution of missing values across all columns is checked.

    Data Preprocessing and Feature Engineering:

        Missing Value Imputation: Missing values in categorical columns (workclass, occupation, native_country) are filled using the mode (the most frequent value).

        Target Variable Encoding: The target column income is converted from categorical (<=50K, >50K) to a binary numerical format (0, 1).

        Feature Transformation: All remaining categorical object columns are converted into numerical format using one-hot encoding.

        Redundant Feature Removal: The education column (which is redundant due to education_num) and fnlwgt are dropped.

    Outlier Detection and Removal:

        Numerical features are analyzed for outliers using box plots.

        The Interquartile Range (IQR) method is applied to identify and remove outlier data points, enhancing model robustness.

        The effect of this step is visualized with "before" and "after" box plots.

    Data Splitting and Scaling:

        The cleaned dataset is split into training (80%) and testing (20%) sets. Stratification is used to ensure the proportion of income classes is the same in both splits.

        Features are scaled using StandardScaler to standardize their ranges, which is crucial for the performance of many ML algorithms.

    Model Training:

        A RandomForestClassifier is instantiated and trained on the preprocessed, scaled training data.

        The model's Out-of-Bag (OOB) score is calculated as a reliable internal measure of performance.

    Model Evaluation:

        The trained model's predictive performance is evaluated on the unseen test set using the Accuracy Score.

        A Confusion Matrix is generated and visualized to provide a detailed breakdown of prediction results, showing how the model performs on both the >50K and <=50K classes.

    Feature Importance Analysis:

        The intrinsic feature importance scores from the Random Forest model are extracted and visualized. This final step helps identify which socio-economic factors were most influential in predicting income level.

Technologies Used

    Python 3

    Pandas (for data manipulation)

    NumPy (for numerical operations)

    Scikit-learn (for modeling, preprocessing, and evaluation)

    Matplotlib & Seaborn (for data visualization)

    Jupyter Notebook / Google Colab (as the development environment)

How to Run This Project

    Prerequisites: Ensure you have a Python environment with the libraries listed above installed.

    Download Files:

        The project notebook (.ipynb file).

        The adult.csv dataset from the Kaggle link provided above.

    Setup Environment:

        Open the notebook file in Google Colab or a local Jupyter Notebook instance.

        Upload the adult.csv file to the same environment.

    Execute Code: Run the notebook cells sequentially from top to bottom to replicate the entire analysis and model training process.

Results and Conclusion

The final model achieves a high accuracy score in predicting whether an individual's income is above or below $50K. The confusion matrix demonstrates the model's effectiveness in correctly classifying individuals in both income brackets.

The feature importance analysis reveals that factors such as age, capital gains, education level (education_num), and hours worked per week are among the most significant predictors of income, providing valuable insights into the key determinants of economic standing.
