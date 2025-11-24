import pandas as pd

melb_path = "melb_data.csv"
melb_data = pd.read_csv(melb_path)

filtered_melb_data = melb_data.dropna(axis=0)

# Choose target and features
y = filtered_melb_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melb_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# define model
melb_model = DecisionTreeRegressor()
# fit model
melb_model.fit(X, y)

predicted_home_prices = melb_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
melb_model = DecisionTreeRegressor()
# Fit Model
melb_model.fit(train_X, train_y)


val_predictions = melb_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
