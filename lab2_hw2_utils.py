import pandas as pd  
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import numpy as np


# converting the txt file dataset tp csv for convenience
def convert_txt_dataset_file_to_csv(text_file_path, csv_file_path):
# Define paths for your files
# text_file_path = 'household_power_consumption.txt'
# csv_file_path = 'household_power_consumption.csv'

    # Open the text file for reading and CSV file for writing
    with open(text_file_path, 'r') as text_file, open(csv_file_path, 'w') as csv_file:
        # Iterate through each line in the text file
        for line in text_file:
            # Replace semicolons with commas (or any other delimiter as needed)
            csv_line = line.replace(';', ',')
            # Write the modified line to the CSV file
            csv_file.write(csv_line)




# utils functions for calculating and plotting the data

def calc_active_energy_consumed_per_hour_in_watt_hour(dataframe):
    df = dataframe.copy()  # Make a copy of the original DataFrame
    
    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with NaN values in the important columns
    df.dropna(subset=['Global_active_power'], inplace=True)

    # Convert 'Date' column to datetime format (parsing dd/mm/yy)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Convert Global_active_power to float
    df['Global_active_power'] = df['Global_active_power'].astype(float)

    # Calculate power consumption per minute in watt-hours (Wh)
    df['Global_active_power_Wh'] = df['Global_active_power'] * 1000 / 60

    # Convert watt-hours (Wh) to kilowatt-hours (kWh)
    df['Global_active_power_kWh'] = df['Global_active_power_Wh'] / 1000

    # Set the Datetime column as the index
    df.set_index('Date', inplace=True)

    # Resample the data to daily frequency and sum the power consumption
    daily_consumption_kWh = df['Global_active_power_kWh'].resample('D').sum()
    weekly_consumption_kWh = df['Global_active_power_kWh'].resample('W').sum()
    monthly_consumption_kWh = df['Global_active_power_kWh'].resample('M').sum()

    return daily_consumption_kWh, weekly_consumption_kWh, monthly_consumption_kWh

def plot_power_consumption_period(data_to_plot, period:str):

    # Plot the daily power consumption in kilowatt-hours (kWh)
    plt.figure(figsize=(10, 6))
    data_to_plot.plot()
    plt.title(f'{period.capitalize()} Power Consumption (kWh)')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (kWh)')
    plt.grid(True)
    plt.show()

def plot_hourly_power_consumption(hourly_consumption):
    plt.figure(figsize=(12, 6))
    plt.hist(hourly_consumption, bins=24, color='blue', alpha=0.7)
    plt.title('Hourly Power Consumption Distribution')
    plt.xlabel('Power Consumption (Wh)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()



def calculate_avg_power_consumption_per_hour(dataframe):
    df = dataframe.copy()
    # Extract hour from 'Time' column (assuming 'Time' is in format "HH:MM:SS")
    df['Hour'] = df['Time'].str.slice(0, 2).astype(int)
    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with NaN values in the important columns
    df.dropna(subset=['Global_active_power'], inplace=True)

    # Convert Global_active_power to float
    df['Global_active_power'] = df['Global_active_power'].astype(float)

    # Calculate average power consumption per hour of each day
    avg_hourly_consumption = df.groupby('Hour')['Global_active_power'].mean().reset_index()

    return avg_hourly_consumption

def plot_hourly_histogram(avg_hourly_consumption):
    # Plotting the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(avg_hourly_consumption['Hour'], avg_hourly_consumption['Global_active_power'], width=0.8, align='center')

    # Customize the plot
    plt.title('Average Power Consumption per Hour 2006-2010')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Global Active Power (kWh)')
    plt.xticks(range(24))
    plt.grid(True)

    # Show the plot
    plt.show()


# def plot_predicted_values(original_data, predicted_values):
#     # Get the last 47 actual values
#     last_actual_values = original_data

#     # Combine actual and predicted values for plotting
#     plot_values = np.concatenate([last_actual_values, predicted_values])
#     # Create an array of indices for x-axis
#     x_values = np.arange(len(plot_values))

#     # Plot the actual values
#     plt.scatter(x_values, plot_values, label='Actual', color='blue')

#     # Plot the predicted values
#     plt.scatter(x_values[-3], plot_values[-3], label='Predicted', color='red', marker='o')

#     # Add labels and title
#     plt.xlabel('Time')
#     plt.ylabel('Global Active Power')
#     plt.title('Global Active Power: Actual vs Predicted')
#     plt.legend()

#     # Show the plot
#     plt.show()


def plot_predicted_values(original_data, predicted_values):
    # Combine actual and predicted values for plotting
    plot_values = np.concatenate([original_data, predicted_values])
    
    # Create an array of indices for x-axis
    x_values = np.arange(len(plot_values))

    # Plot the actual values
    plt.scatter(x_values[:len(original_data)], plot_values[:len(original_data)], label='Actual', color='blue')

    # Plot the predicted values
    plt.scatter(x_values[len(original_data):], plot_values[len(original_data):], label='Predicted', color='red', marker='o')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Global Active Power')
    plt.title('Global Active Power: Actual vs Predicted')
    plt.legend()

    # Show the plot
    plt.show()


def plot_all_metrices(models, metrics):
    # Define the positions for each bar
    x = np.arange(len(models))

    # Set the width of the bars
    width = 0.2

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot MAE
    axs[0, 0].bar(x, metrics['MAE'], width, label='MAE')
    axs[0, 0].set_title('Mean Absolute Error (MAE)')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(models, rotation=45, ha="right")

    # Plot MSE
    axs[0, 1].bar(x, metrics['MSE'], width, label='MSE', color='orange')
    axs[0, 1].set_title('Mean Squared Error (MSE)')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(models, rotation=45, ha="right")

    # Plot RMSE
    axs[1, 0].bar(x, metrics['RMSE'], width, label='RMSE', color='green')
    axs[1, 0].set_title('Root Mean Squared Error (RMSE)')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(models, rotation=45, ha="right")

    # Plot R²
    axs[1, 1].bar(x, metrics['R²'], width, label='R²', color='red')
    axs[1, 1].set_title('R²')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(models, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()
    plt.show()