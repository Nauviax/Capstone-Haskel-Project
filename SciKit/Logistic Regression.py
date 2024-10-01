# Imports
import pandas as pd
import time
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
df = pd.read_csv("../CSV/spambaseCSV.csv")
y_label = df.columns[-1]
X = df.drop(columns=y_label)
y = df[y_label]

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Time from start to finish, excluding loading dataset
time_start_regression = time.perf_counter()

# Create logistic regression object
log_reg = LogisticRegression()

# Train the model using the training sets
log_reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = log_reg.predict(X_test)
mem_end = memory_usage_psutil()

# End time
time_end = time.perf_counter()
time_dataset = time_end - time_start_dataset
time_regression = time_end - time_start_regression
memory_used = mem_end - mem_start

# Results
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f'Time taken, including dataset: {time_dataset:0.4f} seconds')
print(f'Time taken, excluding dataset: {time_regression:0.4f} seconds')
print(f"Memory used: {memory_used:0.2f} MB")