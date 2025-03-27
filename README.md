# AirlinePassengerSatisfaction

#### Overview:
This project focuses on analyzing the satisfaction levels of airline passengers using machine learning models. It aims to predict whether passengers are likely to be satisfied or dissatisfied based on various features such as age, travel type, class, and flight delay times. The workflow includes data preprocessing, visualization, model training, hyperparameter optimization, and evaluation.

---

#### Dataset:
The project uses two datasets:
1. `train.csv`: Training data containing passenger satisfaction details.
2. `test.csv`: Test data to validate model performance.

These files are combined into a single dataset to better utilize data during analysis.

---

#### Workflow:
1. **Loading and Preprocessing the Data:**
   - Data analysis is performed to check for null values, duplicates, and overall data structure.
   - Missing values are imputed using statistical methods (e.g., median for numerical and constant for categorical features).
   - Numerical features are scaled using `StandardScaler`.
   - Categorical features are encoded using `OneHotEncoder`.

2. **Visualization and Data Analysis:**
   - Target distribution analysis using pie and bar charts.
   - Passenger demographic analysis based on gender, customer type, and age distributions.
   - Heatmaps to identify correlations between features and satisfaction levels.
   - Distribution analysis of satisfaction levels based on travel type and flight class.
   - Density plots of passengers’ age against satisfaction levels.

3. **Modeling:**
   Three machine learning models are implemented to predict passenger satisfaction:
   - **Random Forest Classifier**
   - **Gradient Boosting Classifier**
   - **Voting Classifier (Ensemble Model):** Combines Logistic Regression, Random Forest, and XGBoost.

   Each model is optimized using **RandomizedSearchCV** for hyperparameter tuning.

4. **Evaluation:**
   - Models are evaluated using **classification reports**, **confusion matrices**, and **accuracy metrics**.
   - Visualization of model-specific confusion matrices is included for detailed comparison.

---

#### Key Libraries:
The project utilizes the following Python libraries:
- **Data Manipulation:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `sklearn`, `xgboost`, `category_encoders`
- **Warnings Suppression:** `warnings`

---

#### Preprocessing Steps:
1. **Numerical Feature Transformation:**
   - Imputation of missing values using median strategy.
   - Scaling using `StandardScaler`.
   
2. **Categorical Feature Transformation:**
   - Imputation of missing values using constant values.
   - Encoding with `OneHotEncoder`.

3. **Column Transformer and Pipeline:**
   - Features are preprocessed using `ColumnTransformer`, combined into a pipeline along with the classifier.

---

#### Models and Hyperparameter Optimization:
Each model undergoes hyperparameter tuning using **RandomizedSearchCV**:
- **Random Forest:**
  - Parameters such as `n_estimators`, `max_depth`, `criterion`, and `max_features` are optimized.
  
- **Gradient Boosting:**
  - Parameters like `n_estimators`, `learning_rate`, `max_depth`, and `subsample` are searched.

- **Voting Classifier (Ensemble):**
  - Combines `Logistic Regression`, `Random Forest`, and `XGBoost`, with respective hyperparameter tuning.

---

#### Outputs:
1. Best hyperparameters for each model.
2. Predictions on testing data.
3. Confusion matrices for Random Forest, Gradient Boosting, and Ensemble models.
4. Classification reports showcasing precision, recall, and F1 scores.


