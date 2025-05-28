# Predicting-Free-Trial-Conversion-for-Enhanced-Customer-Acquisition-and-Revenue-Growth

# Predicting Free Trial Conversion for Enhanced Customer Acquisition and Revenue Growth

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Key Features and Analysis Steps](#key-features-and-analysis-steps)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to develop a machine learning model to predict whether a free trial user will convert into a paying customer. By accurately identifying potential converters, businesses can optimize their marketing strategies, allocate resources more efficiently, and enhance overall customer acquisition and revenue growth.

## Business Problem
The core business problem addressed is the inefficient allocation of marketing resources and the missed opportunities to convert free trial users. Without a predictive model, businesses may:
- Spend marketing budget on users unlikely to convert.
- Fail to engage high-potential users effectively during their trial period.
- Experience lower conversion rates and suboptimal customer acquisition costs.

This project seeks to provide a data-driven solution to these challenges by predicting conversion likelihood.

## Dataset
The project utilizes a dataset named `digital_marketing_campaign_dataset.csv`, which contains 8000 entries and 20 features related to customer demographics, marketing campaign interactions, and website behavior.

### Key Columns:
- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer.
- **Income**: Customer's income.
- **CampaignChannel**: Channel through which the campaign was delivered (e.g., Social Media, Email, PPC).
- **CampaignType**: Type of marketing campaign (e.g., Awareness, Retention, Conversion).
- **AdSpend**: Amount spent on advertising for the customer.
- **ClickThroughRate**: Rate at which users click on an advertisement.
- **ConversionRate**: Rate at which users complete a desired action after clicking an ad.
- **WebsiteVisits**: Number of visits to the website.
- **PagesPerVisit**: Average number of pages viewed per visit.
- **TimeOnSite**: Average time spent on the website.
- **SocialShares**: Number of times content was shared on social media.
- **EmailOpens**: Number of marketing emails opened.
- **EmailClicks**: Number of clicks within marketing emails.
- **PreviousPurchases**: Number of previous purchases made by the customer.
- **LoyaltyPoints**: Loyalty points accumulated by the customer.
- **AdvertisingPlatform**: Platform used for advertising.
- **AdvertisingTool**: Tool used for advertising.
- **Conversion**: Target variable (1 for converted, 0 for not converted).

### Dataset Characteristics:
- No missing values were found.
- The Conversion target variable shows an imbalanced distribution (approximately 87.65% converted, 12.35% not converted), which is addressed during modeling.

## Key Features and Analysis Steps
The Jupyter Notebook (`Predicting Free Trial Conversion for Enhanced Customer Acquisition and Revenue Growth (1).ipynb`) covers the following steps:

### Data Loading and Initial Exploration:
- Loads the dataset into a Pandas DataFrame.
- Displays basic information (`df.info()`, `df.head()`).
- Generates descriptive statistics (`df.describe()`).
- Checks for missing values (`df.isnull().sum()`).
- Analyzes the distribution of the target variable (`df['Conversion'].value_counts()`).

### Exploratory Data Analysis (EDA):
- Visualizes the distribution of Age and Income using histograms.
- Analyzes Conversion rates across different Gender and CampaignChannel categories using count plots.
- Generates a correlation matrix for numerical features to understand relationships.

### Data Preprocessing:
- Separates features (`X`) and target (`y`).
- Identifies numerical and categorical features.
- Applies `StandardScaler` for numerical features and `OneHotEncoder` for categorical features using `ColumnTransformer`.

### Model Training and Evaluation:
- Splits the data into training and testing sets.
- Addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) within an `ImbPipeline`.
- Trains and evaluates three classification models:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
- Evaluates models using:
  - classification_report (Precision, Recall, F1-score)
  - confusion_matrix
  - roc_auc_score
  - Accuracy

### Hyperparameter Tuning:
- Performs GridSearchCV to find the best hyperparameters for the Random Forest Classifier, optimizing for roc_auc.

### Model Persistence:
- Saves the best-performing trained model (Random Forest Classifier) using joblib for future use.

### Prediction on New Data:
- Demonstrates how to load the saved model and make predictions on new, unseen customer data.
- Outputs the conversion probability and the predicted conversion outcome (Yes/No).

## Technologies Used
- Python: Programming language
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib: Data visualization
- seaborn: Enhanced data visualization
- scikit-learn (sklearn): Machine learning algorithms, preprocessing, and model evaluation
- imbalanced-learn (imblearn): Handling imbalanced datasets (SMOTE)
- joblib: Model persistence

## Installation
To set up the project locally, follow these steps:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

##  Alternatively, create a requirements.txt file with the above libraries and install using:

pip install -r requirements.txt

Usage
Download the dataset: Ensure digital_marketing_campaign_dataset.csv is in the same directory as the notebook, or update the path in the notebook.

Run the Jupyter Notebook: Open the notebook file using Jupyter Notebook or JupyterLab.

jupyter notebook

Execute cells sequentially to perform data loading, EDA, preprocessing, model training, evaluation, and prediction.

Modify the new customer DataFrame at the end of the notebook to test different prediction scenarios.

Results
The project successfully builds and evaluates machine learning models for predicting free trial conversion. The best model (Random Forest Classifier after hyperparameter tuning) is saved for quick predictions on new customer data.

Example output for a new customer prediction:

Prediction for new customer:
Conversion Probability: 89.92%
Predicted Conversion: Yes/No


(where 89.92% and Yes/No are the actual output from the model)

###Contributing
Contributions are welcome! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Make your changes.

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeature).

Open a Pull Request.

### License ###
This project is licensed under the MIT License - see the LICENSE file for details.


