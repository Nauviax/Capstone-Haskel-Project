# Imports
import pandas as pd
import time
import psutil
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score

# Function to measure memory usage
def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    return mem

# Time from start to finish, including loading dataset
time_start_dataset = time.perf_counter()
mem_start = memory_usage_psutil()

# Load the dataset
#df = pd.read_csv('diabetesCSV.csv')
df = pd.read_csv("../CSV/house8L_CSV.csv")
y_label = df.columns[-1]
X = df.drop(columns=y_label)
# X = df[["S1", "S2", "S3"]] # Double brackets to keep 2D
y = df[y_label]

# Create train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) 

# Time from start to finish, excluding loading dataset
time_start_regression = time.perf_counter()

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
mem_end = memory_usage_psutil()

# End time
time_end = time.perf_counter()
time_dataset = time_end - time_start_dataset
time_regression = time_end - time_start_regression
memory_used = mem_end - mem_start

# Results
print('Weights: ')
print(regr.coef_)
print('Intercept: ')
print(regr.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R^2: %.3f" % r2_score(y_test, y_pred))
print(f'Time taken, including dataset: {time_dataset:0.4f} seconds')
print(f'Time taken, excluding dataset: {time_regression:0.4f} seconds') # !!! Just use one time?
print(f"Memory used: {memory_used:0.2f} MB")