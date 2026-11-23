Customer Churn Prediction

1. Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company based on historical customer data.

The project uses Logistic Regression to identify potential churn customers and helps businesses understand customer retention patterns.


2. Project Objectives
- Predict customer churn using Machine Learning.
- Perform data preprocessing and feature engineering.
- Handle missing values and categorical features.
- Improve model performance through threshold tuning.
- Evaluate the model using classification metrics.
- Build a modular and maintainable ML project structure.


3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Pytest


4. Project Structure
classification/
├── data/                       # Dataset files
├── models/                     # Saved ML models
├── results/                    # Model evaluation results
├── src/
│   └── churn_prediction/
│       ├── __init__.py
│       ├── config.py            # Project configuration
│       ├── data_loader.py       # Dataset loading
│       ├── evaluate.py          # Model evaluation
│       ├── exception.py         # Custom exceptions
│       ├── logger.py            # Logging configuration
│       ├── predict.py           # Churn prediction
│       ├── preprocessing.py     # Data preprocessing
│       └── train.py             # Model training
├── test/
│   ├── test_data_loader.py
│   └── test_preprocessing.py
├── main.py                      # Main execution file
├── requirement.txt              # Project dependencies
└── README.md                    # Project documentation


5. Machine Learning Workflow

1. Data Collection
2. Data Loading
3. Exploratory Data Analysis
4. Data Preprocessing
5. Missing Value Handling
6. Categorical Feature Encoding
7. Feature Scaling
8. Train-Test Split
9. Logistic Regression Model Training
10. Threshold Tuning
11. Model Evaluation


6. Data Preprocessing
The preprocessing pipeline includes:
- Removing unnecessary customer identifiers.
- Handling missing numerical values using median imputation.
- Handling missing categorical values using most frequent imputation.
- Applying One-Hot Encoding to categorical features.
- Applying StandardScaler to numerical features.
- Splitting data into training and testing sets using stratification.

7. Model Used
Logistic Regression
Logistic Regression is used as the initial classification algorithm to predict whether customers are likely to churn.
Class balancing and classification threshold tuning are used to improve the model's ability to identify churn customers.

8. Model Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Latest Experimental Results
Metric	    Score
Accuracy	70.33%
Precision	46.62%
Recall	    83.11%
F1-Score	59.73%


Classification Threshold: 0.30


9. Installation
1. Clone the repository:
2. git clone YOUR_GITHUB_REPOSITORY_URL
3. Navigate to the project directory:
4. cd classification
5. Create a virtual environment:
python3 -m venv .venv
Activate the environment:
source .venv/bin/activate
Install dependencies:
pip install -r requirement.txt

10. Running the Project
Run the main execution file:
python main.py

11. Running Tests
Run the test cases using:
pytest test/

12. Future Improvements
- Experiment with Random Forest and XGBoost.
- Improve the precision and recall balance.
- Perform feature selection.
- Further optimize the classification threshold.
- Compare multiple classification algorithms.

13. Author
Sanket Kishor Rane