# Experiment with Neptune.ai Group A
# Importing pandas for data manupulation
import pandas as pd
# Importing pickle for model exporting
import pickle
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math 
## Initialize Neptune
import neptune.new as neptune
from neptune.new.types import File
import neptune.new.integrations.sklearn as npt_utils
parameters_rfr = {}
# ------------  Code Below -----------------------
# Initilise neptune
run = neptune.init(project='',
                   api_token='')
# ------------  Code Above -----------------------
#-------------------- Preparing the Data -------------------------
## Uploaing Dataset
data = load_boston()
y = data.target

# Converting data to DataFrame
df = pd.DataFrame(data=data['data'], columns = data['feature_names'])

# Feature Selection
#df = df[['NOX', 'CHAS', 'INDUS', 'DIS']]  #Feature 1
#df = df[['PTRATIO', 'TAX', 'RM']]   # Feature 2
# Here we are downloading the DataFrame as .csv file
dataset = df.to_csv("data.csv", index=None)

# Between the quotes write a path where you want to save the data.csv. e.g 'dataset'
#run['dataset'].upload("data.csv")
    
#-------------------- The Parameters -------------------------

# The parameters for Random Forest Regressor (RFR)
#parameters_rfr = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 6, 'ccp_alpha':0.1}

# Splitting the data and setting test_size
split_param = {'test_size': 0.20, 'random_state': 28750}

## Log the parameters and split_param below and between the quotes write a path where you want to save the data.csv. e.g 'parameter/parameters_rfr':
# ------------  Write Code Below (Track Parameters Here) -----------------------
#run["parameter/parameters_rfr"] = parameters_rfr
#run["parameter/split_param"] = split_param
# ------------  Write Code Above -----------------------

# Write Code Below (Download the model)
#run["model"].download()

#with open('model.pkl', 'rb') as f:
#    model = pickle.load(f)

#-------------------- The Model -------------------------
#model = RandomForestRegressor(**parameters_rfr)
model = LinearRegression(normalize=True)
 
X_train, X_test, y_train, y_test = train_test_split(df, y, **split_param)

# Fitting the model
model.fit(X_train, y_train)



# ------------  Code Below (Track Model Here)-----------------------
#run["model"] = npt_utils.get_pickled_model(model)
# ------------  Write Code Above -----------------------

# Here we are getting the Evaluation Metrics for: Training and Test Set.

# Test Set
y_test_predict = model.predict(X_test)
mean_absolute_error = mean_absolute_error(y_train, y_test_predict)
rmse = (math.sqrt(mean_squared_error(y_train, y_test_predict)))
r2 = r2_score(y_train, y_test_predict)

# ------------  Write Code Below (Track Test Evaluation Metrics) -----------------------
#run['scores/training_set/rmse'] = rmse
#run['scores/training_set/mean_absolute_error'] = mean_absolute_error
#run['scores/training_set/r2_score'] = r2
# ------------  Write Code Above ----------------------- 

# Write solutions to task below: