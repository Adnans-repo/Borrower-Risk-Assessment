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

```python
# Oversample the minority class using SMOTE
smote = SMOTE(sampling_strategy="minority", random_state=0)
features_train, target_train = smote.fit_resample(features_train, target_train)
# print the elapsed time
print(f'\nDuration: {time.time() - start_time:.0f} seconds') 
```

# Evaluation Metrics Criteria
**F1 Score:** The F1 Score is a weighted average of precision and recall, and is used to evaluate the performance of the classifier. A higher F1 score indicates that the classifier is able to accurately classify both positive and negative samples.

### F1-Score = 2 * Precision * Recall / (Precision + Recall)
Apart from this we'll also be using ROC-AUC and classification accuracy metrics

# Feature Engineering: Enhancing Features to Improve Model Accuracy
This process involves selecting and modifying features to create a more informative representation of the data. Our feature engineering approach includes:

  1. Correlation Analysis: Identifying the strongest relationships between features and the target variable.
  2. Feature Selection: Choosing only the most relevant features to include in the model.
  3. Feature Scaling: Standardizing feature values to ensure equal weighting in the model.

By refining our features through these methods, we aim to create a model that predicts credit risk with greater precision and accuracy.

### Normalizing and Scaling for Optimal Model Performance
```python
def normalize(subset):
    continuous_columns = subset.select_dtypes(include=['float']).columns
    scaler = preprocessing.MinMaxScaler()
    subset[continuous_columns] = scaler.fit_transform(subset[continuous_columns])
    return subset
```
### Understanding Feature Relationships through Correlation Analysis
Correlation analysis helps us assess the strength of relationships between features and the target variable. By identifying which features are most strongly correlated with the target, we can make informed decisions about which features to include in our model.

![cr1](https://user-images.githubusercontent.com/116202234/217828813-41e8318e-ff50-4ac4-bd7d-fa9f52047ad5.png)

# Model Comparison and Selection
In order to build an accurate credit risk analysis model, it's important to evaluate the performance of different models. We will be testing a variety of suitable models and comparing their accuracy to determine which one is best suited for our project.

Some of the models we will be evaluating include Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machines (SVMs), and Neural Networks. These models will be trained and tested on our dataset, and the accuracy of each model will be evaluated.

Based on the results of this comparison, we will select the model with the highest accuracy for our final implementation. By conducting a thorough evaluation of different models, we can ensure that we are using the best possible model for our credit risk analysis project.

