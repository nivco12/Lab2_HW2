import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


def prepare_rnn_data(df, target_col, lags):
    # Ensure only numeric columns are used
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Normalize the dataset
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

    # Create sequences
    X, y = create_sequences(df_scaled, target_col, lags)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

def create_sequences(df, target_col, lags):
    data = df.values
    target_index = df.columns.get_loc(target_col)
    
    num_samples = len(df) - lags
    num_features = df.shape[1]
    
    X = np.zeros((num_samples, lags, num_features))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        X[i] = data[i:i+lags]
        y[i] = data[i+lags, target_index]
    
    return X, y

def build_and_train_rnn(X_train, y_train, lags, num_units=25, epochs = 5, batch_size = 64):
    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=num_units, input_shape=(lags, X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def evaluate_model(model, X_test, y_test, scaler, target_col):
    # Predict with the model
    y_pred = model.predict(X_test)

    # Ensure y_pred is 2D for concatenation
    y_pred = y_pred.reshape(-1, 1)

    # Create zero padding for non-target features
    padding = np.zeros((y_pred.shape[0], X_test.shape[2] - 1))

    # Concatenate padding with predictions
    concatenated_pred = np.concatenate((padding, y_pred), axis=1)

    # Inverse transform the predictions
    y_pred_inv = scaler.inverse_transform(concatenated_pred)[:, -1]

    # Similarly, inverse transform the test target values
    y_test_reshaped = y_test.reshape(-1, 1)
    concatenated_test = np.concatenate((padding, y_test_reshaped), axis=1)
    y_test_inv = scaler.inverse_transform(concatenated_test)[:, -1]

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared (R²) value: {r2:.3f}")

    return mae, mse, rmse, r2  


def predict_next_values_rnn(model, recent_data, scaler, lags, num_predictions=3):
    # Prepare the list to store predictions
    predictions = []

    for _ in range(num_predictions):
        # Predict the next value
        predicted_value = model.predict(recent_data)
        
        # Append the prediction to the list
        predictions.append(predicted_value[0, 0])
        
        # Update the input data with the new prediction
        # Remove the first time step and append the predicted value at the end
        recent_data = np.roll(recent_data, -1, axis=1)
        recent_data[0, -1, 0] = predicted_value  # Update only the target column

    # Inverse transform the predictions
    # Create zero padding for non-target features
    padding = np.zeros((len(predictions), recent_data.shape[2] - 1))
    
    # Concatenate padding with predictions
    predictions_with_padding = np.concatenate((padding, np.array(predictions).reshape(-1, 1)), axis=1)
    
    # Inverse transform the predictions
    predictions_inv = scaler.inverse_transform(predictions_with_padding)[:, -1]
    
    return predictions_inv
