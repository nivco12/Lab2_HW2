from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Function to create lag features
def create_lag_features(df, target_col, lags):
    df_copy = df.copy()
    for lag in range(1, lags + 1):
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
    df_copy.dropna(inplace=True)
    return df_copy


# Function to train and evaluate linear regression model
def train_and_evaluate_lagged_model(df, target_col, lags):
   
    # copy the dataset into a new object
    df_lagged = df.copy()
    
    # Create lag features
    df_lagged = create_lag_features(df, target_col, lags)
    
    # Define features (X) and target (y)
    X = df_lagged[[f'{target_col}_lag_{i}' for i in range(1, lags + 1)]]
    y = df_lagged[target_col]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Return the model and the predictions
    return model, y_pred, y_test



# Evaluate performance for each model
def evaluate_performance(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2
