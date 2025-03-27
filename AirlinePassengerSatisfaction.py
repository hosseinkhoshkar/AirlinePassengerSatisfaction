#Math
import numpy as np
import pandas as pd 

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import cm

#ML
import sklearn
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

#Data transformation
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Warnings
import warnings
warnings.filterwarnings("ignore")

#Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV, train_test_split

#Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Plot colors
target_colors=['#f44f49', '#49eef4']

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
		
        
        

##Loading the Dataset
df_train = pd.read_csv('C:/Users/Hossein/Documents/train.csv')
df_test = pd.read_csv('C:/Users/Hossein/Documents/test.csv')

df = pd.concat([df_train, df_test])

df.head()

print(df.shape)
print(df_train.shape)
print(df_test.shape)

## dataset overview
df.info()

#A simpler way to find missing values
df.isna().sum() 

## Interpretation and drow charts 
fig = plt.figure(figsize=(22, 6))
plt.suptitle('Target Distribution', weight='bold', fontsize=24, fontname='Arial')  

grid = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)

ax1 = fig.add_subplot(grid[0, :1])
ax1.set_title('Satisfaction Count')  

sns.countplot(x='satisfaction', data=df, ax=ax1, palette=target_colors)


for spine in ax1.spines.values():
    spine.set_visible(False)

ax1.get_yaxis().set_visible(False)

for index,value in enumerate(df['satisfaction'].value_counts()):
    ax1.annotate(value,xy=(index,value+2000), ha='center', va='center', fontsize=15)

ax1.set_xticklabels(df['satisfaction'].value_counts().index, fontsize=15)

#pie plot

ax2=fig.add_subplot(grid[0, 1:])
ax2.set_title('Target Weight')
label=list(df['satisfaction'].value_counts().index)
value=list(df['satisfaction'].value_counts().values)
#pie chart
ax2.pie(value, labels=label, autopct='%1.1f%%', explode=(0,0.2), startangle =90, colors =target_colors)

plt.show()

## get passengers profile and drow charts 
#gender, customertype , class und age

fig = plt.figure(figsize=(30, 18))
plt.suptitle('Passenger Profile', weight='bold', fontsize=24, fontname='Arial') 

grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

ax1 = fig.add_subplot(grid[0, :1])
ax1.set_title('Gender', fontsize=18)  

label=list(df['Gender'].value_counts().index)
value=list(df['Gender'].value_counts().values)
#pie chart
ax1.pie(value, labels=label, autopct='%1.1f%%', explode=(0,0.2), startangle =90, colors =target_colors)


ax2 = fig.add_subplot(grid[0,1:])
ax2.set_title('Customer Type', fontsize=18)

label=list(df['Customer Type'].value_counts().index)
value=list(df['Customer Type'].value_counts().values)
#pie chart
ax2.pie(value, labels=label, autopct='%1.1f%%', explode=(0,0.2), startangle =90, colors =target_colors)


ax3 = fig.add_subplot(grid[1,:1])
ax3.set_title('Class', fontsize=18)  

label=list(df['Class'].value_counts().index)
value=list(df['Class'].value_counts().values)
#pie chart
ax3.pie(value, labels=label, autopct='%1.1f%%', startangle =90, colors =target_colors+['#324DBB'])


ax4 = fig.add_subplot(grid[1,1:])
ax4.set_title('Age of Passengers', fontsize=18)  

sns.kdeplot(data=df, x='Age', ax=ax4, fill=True)

ax4.tick_params(axis='x',labelsize = 20)
ax4.tick_params(axis='y',labelsize = 20)


ax4.set_xlabel('Age of Passengers', fontsize=20, weight ='bold')
ax4.set_ylabel('Density', fontsize=20, weight ='bold')

for spine in ax4.spines.values():
    spine.set_visible(False)

ax4.axvline(df['Age'].mean(), linestyle='--', color='#324DBB')
ax4.legend(fontsize=20)

plt.show()


fig = plt.figure(figsize=(30, 18))
plt.suptitle('Passenger Distribution', weight='bold', fontsize=24, fontname='Arial') 

grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

ax1 = fig.add_subplot(grid[0, :1])
ax1.set_title('Gender Distribution', fontsize=18) 


sns.countplot(x=df['Gender'], hue=df['satisfaction'], ax=ax1, palette=target_colors)

for p in ax1.patches:
    ax1.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.1,p.get_height()), fontsize=20)

ax1.get_yaxis().set_visible(False)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.tick_params(axis='x', labelsize=15)
ax1.set_xlabel('Gender', fontsize=20)


grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

ax2 = fig.add_subplot(grid[0, 1:])
ax2.set_title('Class Distribution', fontsize=18) 

sns.countplot(x=df['Class'], hue=df['satisfaction'], ax=ax2, palette=target_colors)

for p in ax2.patches:
    ax2.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.05,p.get_height()), fontsize=20)

ax2.get_yaxis().set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.tick_params(axis='x', labelsize=15)
ax2.set_xlabel('Class', fontsize=20)


grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

ax3 = fig.add_subplot(grid[1, :1])
ax3.set_title('Travel Distribution', fontsize=18) 

sns.countplot(x=df['Type of Travel'], hue=df['satisfaction'], ax=ax3, palette=target_colors)

for p in ax3.patches:
    ax3.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.05,p.get_height()), fontsize=20)

ax3.get_yaxis().set_visible(False)
for spine in ax3.spines.values():
    spine.set_visible(False)
ax3.tick_params(axis='x', labelsize=15)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='satisfaction', bins=30, kde=True, palette=target_colors)
plt.title('Age Distribution by Satisfaction', fontsize=18, weight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

#before
df_copy = df.copy()

mapping = {'neutral or dissatisfied': 1, 'satisfied': 0}
df_copy['satisfaction_mapped'] = df_copy['satisfaction'].map(mapping)

num_df = df_copy.drop(columns=['Unnamed: 0', 'id']).select_dtypes(include=['int64', 'float64'])
num_df['satisfaction'] = df_copy['satisfaction_mapped']

correlation_with_satisfaction = num_df.corr()['satisfaction'].sort_values(ascending=False)

correlation_df = correlation_with_satisfaction.to_frame(name='Correlation').iloc[1:]  # Exkludiere 'satisfaction' selbst

plt.figure(figsize=(8, 10))
sns.heatmap(correlation_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation with Satisfaction', fontsize=16)

plt.show()

df_copy = df.copy()

df_copy = df.drop(columns=['id', 'Unnamed: 0'])

df_copy.duplicated().sum() #Keine Duplikate         

#X_train['Arrival Delay in Minutes'].fillna(X_train['Arrival Delay in Minutes'].median(axis = 0), inplace = True)
#X_test['Arrival Delay in Minutes'].fillna(X_test['Arrival Delay in Minutes'].median(axis = 0), inplace = True)

X = df_copy.drop(columns=['satisfaction'])
y = df_copy['satisfaction'].map({"neutral or dissatisfied": 0, "satisfied": 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Splitting the data ratio 80:20

# ce_OHE = ce.OneHotEncoder(cols=['Gender','Customer Type', 'Type of Travel','Class'])
# X_train = ce_OHE.fit_transform(X_train)

# X_train.head()

# X_test = ce_OHE.fit_transform(X_test)
# X_test.head()

#scaler = StandardScaler()

# columns_to_normalize = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
# X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])

numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Create a numerical transformer (Impute + Scale)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Normalize the data
])

# Create a categorical transformer (Impute + OneHotEncode)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    
    
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  
    ('classifier', RandomForestClassifier(random_state=42))  
])
param_dist = {
     'classifier__n_estimators': randint(50, 200),  # Random numbers fro 50-200 are chosen
     'classifier__max_depth': [None, 10, 20, 30],  # Choose from these fixed values
     'classifier__min_samples_split': randint(1, 10),  # Randomly choose between 2 and 10
     'classifier__min_samples_leaf': randint(1, 10),  # Randomly choose between 1 and 5
     'classifier__criterion': ['gini', 'entropy'],  # Criterion for splitting nodes
     'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Choose one of these options
     'classifier__bootstrap': [True, False]  # Choose either True or False
}

# Instantiate RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist, 
    n_iter=10,  # Number of random combinations to try
    cv=5,  # Number of cross-validation folds
    scoring='accuracy',  # Scoring method
    n_jobs=-1,  # Use all available cores
    verbose=1,  # Show progress of search
    random_state=42  # Set random state for reproducibility
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best model
best_model_rf = random_search.best_estimator_

best_params_rf = random_search.best_params_

y_pred_rf = best_model_rf.predict(X_test)

best_model_rf


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  
    ('classifier', GradientBoostingClassifier(random_state=42)) 
])

# Define the parameter distributions for Gradient Boosting
param_dist = {
    'classifier__n_estimators': randint(50, 200),  # Number of boosting stages
    'classifier__learning_rate': uniform(0.01, 0.3),  # Learning rate for shrinkage
    'classifier__max_depth': randint(3, 10),  # Depth of each tree
    'classifier__min_samples_split': randint(2, 10),  # Minimum samples to split a node
    'classifier__min_samples_leaf': randint(1, 5),  # Minimum samples in a leaf
    'classifier__subsample': uniform(0.7, 0.3),  # Fraction of samples used for fitting
    'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Features considered for split
}

# Instantiate RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=5,  # Number of cross-validation folds
    scoring='accuracy',  # Scoring metric
    n_jobs=-1,  # Use all available cores
    verbose=1,  # Show progress
    random_state=42  # For reproducibility
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best model
best_model_gb = random_search.best_estimator_

best_params_gb = random_search.best_params_

# Make predictions using the best model
y_pred_gb = best_model_gb.predict(X_test)


best_model_gb

# Logistic Regression
logistic = LogisticRegression(random_state=42)

# Random Forest
random_forest = RandomForestClassifier(random_state=42)

# Gradient Boosting
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ('lr', logistic),         # Logistic Regression
        ('rf', random_forest),    # Random Forest
        ('xgb', xgb)              # XGBoost
    ],
    voting='soft'  # Use probabilities for voting
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Add preprocessing step
    ('classifier', ensemble)         # Add ensemble classifier
])

param_dist = {
    'classifier__lr__C': uniform(0.1, 10),  # Logistic Regression: Regularization strength
    'classifier__rf__n_estimators': randint(50, 200),  # Random Forest: Number of trees
    'classifier__rf__max_depth': [None, 10, 20, 30],   # Random Forest: Max depth
    'classifier__xgb__n_estimators': randint(50, 200), # XGBoost: Number of trees
    'classifier__xgb__learning_rate': uniform(0.01, 0.3),  # XGBoost: Learning rate
    'classifier__xgb__max_depth': [3, 5, 7]  # XGBoost: Max depth
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations
    cv=5,       # Cross-validation folds
    scoring='accuracy',  # Metric to optimize
    n_jobs=-1,  # Use all available cores
    verbose=1,
    random_state=42
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best model
best_model_ens = random_search.best_estimator_

#Get best params for EVA
best_params_ens = random_search.best_params_

# Predict on test data
y_pred_ens = best_model_ens.predict(X_test)

best_model_ens

print("Classification Report Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Classification Report XGB:")
print(classification_report(y_test, y_pred_gb))
print("Classification Report Combined Model:")
print(classification_report(y_test, y_pred_ens))

fig = plt.figure(figsize=(50, 30))  
plt.suptitle('Confusion Matrix of each model', weight='bold', fontsize=60, fontname='Arial')


grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# First plot 
ax1 = fig.add_subplot(grid[0, 0])
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap=target_colors, ax=ax1,
            annot_kws={"size": 80})  # Larger annotation font size
ax1.set_title('Random Forest', fontsize=50)
ax1.set_xlabel('Predicted Label', fontsize=50)
ax1.set_ylabel('True Label', fontsize=50)

# Second plot 
ax2 = fig.add_subplot(grid[0, 1])
sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True, fmt='d', cmap=target_colors, ax=ax2,
            annot_kws={"size": 80})  # Larger annotation font size
ax2.set_title('Extreme Gradient Boosting', fontsize=50)
ax2.set_xlabel('Predicted Label', fontsize=50)
ax2.set_ylabel('True Label', fontsize=50)

# Third plot
ax3 = fig.add_subplot(grid[1, 0])
sns.heatmap(confusion_matrix(y_test, y_pred_ens), annot=True, fmt='d', cmap=target_colors, ax=ax3,
            annot_kws={"size": 80}) 
ax3.set_title('Combined Model', fontsize=50)
ax3.set_xlabel('Predicted Label', fontsize=55)
ax3.set_ylabel('True Label', fontsize=55)


plt.tight_layout(pad=6.0) 
plt.show()

# Print the best hyperparameters found
print("Best Hyperparameters Random Forest:")
best_params_rf

# Print the best hyperparameters found
print("Best Hyperparameters Extreme Gradient Boosting:")
best_params_gb

# Print the best hyperparameters found
print("Best Hyperparameters Combined Model:")
best_params_ens

