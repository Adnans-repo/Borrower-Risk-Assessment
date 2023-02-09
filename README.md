# Borrower-Risk-Assessment
The goal of this project is to build a predictive model for loan default using machine learning techniques on a historical loan dataset. The model is evaluated with cross-validation and various performance metrics to determine its efficacy in accurately assessing credit risk for loan applications.

## Information on the Data
The data used in this project is a historical loan dataset that includes information about borrower characteristics (e.g., credit score, income, employment history) and loan information (e.g., loan amount, interest rate, default status). The data was obtained from a publicly available source and has been preprocessed to handle missing values and outliers.

**Dataset overview:**

| Label | Description | Instances
| --- | --- | --- | 
|0| Repaid the loan | 33136 |
|1| Defaulted | 5634 | 

Source: [Lending Club 2007](https://www.kaggle.com/datasets/samaxtech/lending-club-20072011-data)

![dataset-2](https://user-images.githubusercontent.com/116202234/217816394-4890c4a8-c27e-4803-b59d-0891c88750ec.PNG)

# Balancing the Imbalanced Dataset with Synthetic Minority Over-sampling Technique (SMOTE)
In this project, we are performing credit risk analysis using a dataset collected from Lending Club in 2007. The dataset has 5634 samples of the default class and 33136 samples of the non-default class, which creates an imbalanced distribution of class labels.

To mitigate this imbalance, we are using the Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic samples of the minority class. SMOTE works by interpolating between existing samples of the minority class to create new, synthetic samples. This helps to balance the distribution of class labels in the dataset and improves the performance of the machine learning model.

By oversampling the minority class using SMOTE, we aim to build a more robust and accurate model for credit risk analysis, which can make better predictions for the default class.
