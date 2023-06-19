# Experiment with Neptune.ai Group C
# Importing pandas for data manupulation
import pandas as pd
# Importing pickle for model exporting
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math 

parameters_rfr = {}

#-------------------- Preparing the Data -------------------------
## Uploaing Dataset
data = fetch_california_housing()
y = data.target

# Converting data to DataFrame
df = pd.DataFrame(data=data['data'], columns = data['feature_names'])

# Feature Selection
#df = df[['MedInc', 'HouseAge', 'AveBedrms']]
# Here we are downloading the DataFrame as .csv file
dataset = df.to_csv("data.csv", index=None)

    
#-------------------- The Parameters -------------------------

# The parameters for Random Forest Regressor
#parameters_rfr = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 6, 'ccp_alpha':0.1}

# Splitting the data and setting test_size
split_param = {'test_size': 0.20, 'random_state': 28750}

#-------------------- The Model -------------------------
#model = RandomForestRegressor(**parameters_rfr)
model = LinearRegression(normalize=True)
 
X_train, X_test, y_train, y_test = train_test_split(df, y, **split_param)

# Fitting the model
model.fit(X_train, y_train)


# Training Set
y_train_predict = model.predict(X_train)
mean_absolute_error = mean_absolute_error(y_train, y_train_predict)
rmse = (math.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)


# Test Set
y_test_predict = model.predict(X_test)
rmse_test = (math.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

# Write solutions to task below:
