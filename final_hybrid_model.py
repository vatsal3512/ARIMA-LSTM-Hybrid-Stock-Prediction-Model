import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()
Filename_address = os.getenv("FILE_ADDRESS")
Output_address = os.getenv("OUTPUT_ADDRESS")
close = "Close"
lag = os.getenv("LAG")
epochs = int(os.getenv("EPOCHS"))
learning_rate = float(os.getenv("LEARNING_RATE"))
batch_size = int(os.getenv("BATCH_SIZE"))
number_nodes = int(os.getenv("NUMBER_NODES"))
days = int(os.getenv("Prediction_days"))
n = int(os.getenv("NN_LAGS"))

# Basically loading the data and making a data-frame wrt to time.
def data_loader():
   cols = ["Close", "High", "Low", "Open", "Volume"]
   data = pd.read_csv(Filename_address, index_col="Date", parse_dates=True)
   data.columns = cols
   data = data.dropna()
   print(f"The Shape of the Data-Set is : {data.shape}\nThe Data-Set is : \n{data.head()}\n")
   return data

# Plotting Line Graph with data and column name 
def plot_predictions(train, predictions,title):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='Actual')
    plt.plot(train.index, predictions, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close-Price')
    address = Output_address + title + ".jpg"
    plt.savefig(address)
    
def plot_raw_data(data):
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data[close], label='Close Price')
    plt.title('Raw Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    address = Output_address + 'Raw Time Series Data' + ".jpg"
    plt.savefig(address)
    
def plot_train_test(train, test):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='Train Set')
    plt.plot(test.index, test, label='Test Set', color='orange')
    plt.title('Train and Test Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    address = Output_address + 'Train and Test Data' + ".jpg"
    plt.savefig(address)
    
def plot_prediction_errors(errors):
    plt.figure(figsize=(10,5))
    plt.plot(errors, label='Prediction Errors')
    plt.title('Prediction Errors over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.legend()
    address = Output_address + 'Prediction Errors over Time' + ".jpg"
    plt.savefig(address)

def plot_final_predictions(test, final_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, final_predictions, label='Corrected Prediction', color='green')
    plt.title('Final Predictions with Error Correction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    address = Output_address + 'Final Predictions with Error Correction' + ".jpg"
    plt.savefig(address)

def plot_accuracy(mse, rmse, mae):
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    plt.figure(figsize=(10,5))
    plt.bar(metrics, values, color=['blue', 'orange', 'green'])
    plt.title('Model Accuracy Metrics')
    address = Output_address + 'Model Accuracy Metrics' + ".jpg"
    plt.savefig(address)

def plot_arima_accuracy(mse, rmse, mae):
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'orange', 'green'])
    plt.title('ARIMA Model Accuracy Metrics')
    address = Output_address + 'Model Accuracy Metrics' + ".jpg"
    plt.savefig(address)
    
        
# Data Partination For my model development and training.
def data_allocation(data):
   train_len_val = len(data) - days
   train,test = data[close].iloc[0:train_len_val],data[close].iloc[train_len_val:]
   print("\n--------------------------------- The Training Set is : -------------------------------------------\n")
   print(train)
   print(f"\nThe Number of Enteries : {len(train)}\n")
   print("\n--------------------------------- The Testing Set is : --------------------------------------------\n")
   print(test)
   print(f"\nThe Number of Enteries : {len(test)}\n")
   return train,test

def apply_transform(data, n: int):
    middle_data = []
    target_data = []
    for i in range(n, len(data)):
        input_sequence = data.iloc[i-n:i]  
        middle_data.append(input_sequence) 
        target_data.append(data.iloc[i])
    middle_data = np.array(middle_data).reshape((len(middle_data), n, 1))
    target_data = np.array(target_data)
    return middle_data,target_data

class PyTorchLSTM(nn.Module):
   def __init__(self, number_nodes):
      super().__init__()
      self.lstm = nn.LSTM(input_size=1, hidden_size=number_nodes, batch_first=True)
      self.fc1 = nn.Linear(number_nodes, number_nodes)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(number_nodes, number_nodes)
      self.fc3 = nn.Linear(number_nodes, 1)

   def forward(self, x):
      out, _ = self.lstm(x)
      out = out[:, -1, :] 
      out = self.relu(self.fc1(out))
      out = self.relu(self.fc2(out))
      out = self.fc3(out)
      return out

class ModelWrapper:
   def __init__(self, pt_model):
      self.model = pt_model
      self.model.eval()
   def predict(self, x):
      x_t = torch.tensor(x, dtype=torch.float32)
      with torch.no_grad():
         return self.model(x_t).numpy()
   def summary(self):
      return "PyTorch LSTM Model (Summary omitted)"

# This the LSTM model training Function 
def LSTM(train,n : int, number_nodes, learning_rate, epochs, batch_size):
   middle_data, target_data = apply_transform(train, n)
   X = torch.tensor(middle_data, dtype=torch.float32)
   y = torch.tensor(target_data, dtype=torch.float32).view(-1, 1)
   
   dataset = TensorDataset(X, y)
   loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
   
   model = PyTorchLSTM(number_nodes)
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   
   history = []
   model.train()
   for epoch in range(epochs):
      epoch_loss = 0
      for batch_X, batch_y in loader:
         optimizer.zero_grad()
         outputs = model(batch_X)
         loss = criterion(outputs, batch_y)
         loss.backward()
         optimizer.step()
         epoch_loss += loss.item()
      history.append(epoch_loss / len(loader))
      
   print(f"middle_data shape: {middle_data.shape}")
   print(f"target_data shape: {target_data.shape}")
   
   model.eval()
   with torch.no_grad():
      full_predictions = model(X).numpy().flatten()
      
   return ModelWrapper(model), history, full_predictions

# Calculating Accuracy of the Both the Models 
def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse,rmse,mae

# Error Evaluation from the Prediction made from LSTM Model.
def Error_Evaluation(train_data,predict_train_data,n:int):
   errors = []
   for i in range(len(predict_train_data)):
      err = train_data.iloc[n + i] - predict_train_data[i]
      errors.append(err)
   return errors

# ARIMA Parameter Selection and PACF & ACF
def Parameter_calculation(data):
   finding = auto_arima(data,trace = True)
   plot_acf(data,lags = lag)
   address = Output_address + "ACF" +".jpg"
   plt.savefig(address)
   plot_pacf(data,lags = lag)
   address = Output_address + "PACF" +".jpg"
   plt.savefig(address)
   ord = finding.order
   return ord

# ARIMA Model Function for Predicting the possible ERRORS from LSTM Model.
def ARIMA_Model(train,len_test,ord):
   model = ARIMA(train, order = ord)
   model = model.fit()
   predictions = model.predict(start = len(train),end = len(train) + len_test ,type='levels')
   full_predictions = model.predict(start = 0,end = len(train)-1,type='levels')
   return model,predictions,full_predictions

# The Final Prediction : LSTM predicted value + ARIMA predicted Error value
def Final_Predictions(predictions_errors,predictions):
   final_values = []
   for i in range(days):
      final_values.append(predictions_errors[i] + predictions[i])
   return final_values

# Main Function
def main():
    data = data_loader() 
    plot_raw_data(data) 
    train, test = data_allocation(data)
    plot_train_test(train, test)
    print(f"Enter the Lag Value for the Neural Network to Work : {n}\n")
    # LSTM Model
    st1 = time.time()
    model, history, full_predictions = LSTM(train, n, number_nodes, learning_rate, epochs, batch_size)
    plot_predictions(train.iloc[n:], full_predictions,"LSTM PREDICTIONS VS ACTUAL Values For TRAIN Data Set")
    last_sequence = train.iloc[-n:].values.reshape((1, n, 1))
    predictions = []
    for i in range(days+1):
        next_prediction = model.predict(last_sequence).flatten()[0]
        predictions.append(next_prediction)
        if i < len(test):
            actual_value = test.iloc[i]
            new_row = np.append(last_sequence[:, 1:, :], np.array([[[actual_value]]]), axis=1)
        else:
            new_row = np.append(last_sequence[:, 1:, :], np.array([[[next_prediction]]]), axis=1)        
        last_sequence = new_row.reshape((1, n, 1))
    plot_predictions(test,predictions[:-1], "LSTM Predictions VS Actual Values")
    errors_data = Error_Evaluation(train,full_predictions,n)
    plot_prediction_errors(errors_data)
    print(f"\n\n----------------------------- THE {days} PREDICTION VALUES FROM LSTM ---------------------------------------------------\n\n")
    for i in range(days):
        actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
        print(f"Day {i+1} => ACTUAL VALUE : {actual_value} | PREDICTED VALUE : {predictions[i]}\n")        
    print("\n---------------------------- The LSTM Model Summary is : ----------------------------\n")
    print(model.summary())
    mse, rmse, mae = calculate_accuracy(test[:days], predictions[:days])
    plot_accuracy(mse, rmse, mae) 
    print("\n----------------------------- LSTM MODEL ACCURACY -----------------------------\n")
    print(f"\nMEAN SQUARED ERROR : {mse}\nROOT MEAN SQUARED ERROR : {rmse}\nMEAN ABSOLUTE ERROR : {mae}\n\n")
    
    
    
    ord = Parameter_calculation(errors_data)
    Arima_Model,predictions_errors,full_predictions_errors = ARIMA_Model(errors_data,len(test),ord)
    print(f"\n\n---------------------------- ARIMA MODEL {days} Predictions-------------------------\n\n")
    for i in range(len(predictions_errors)):
       print(f"{i+1} : {predictions_errors[i]}\n")
    print("\n---------------------------- ARIMA MODEL Summary -------------------------\n")
    print(Arima_Model.summary())
    arima_mse, arima_rmse, arima_mae = calculate_accuracy(errors_data, full_predictions_errors)
    plot_arima_accuracy(arima_mse, arima_rmse, arima_mae)
    
    
    print("\n\n--------------------------- FINAL PREDICTIONS ---------------------------------\n\n")
    final_predictions = Final_Predictions(predictions_errors,predictions)
    plot_final_predictions(test[:days], final_predictions[:days])
    for i in range(days):
       actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
       print(f"Day {i+1} => ACTUAL VALUE : {actual_value} | PREDICTED VALUE : {final_predictions[i]}\n")

    print("\n---------------- Difference Between the LSTM Predictions and Final Predictions of {days} days ----------------\n")
    for i in range(days):
       actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
       print(f"\n{i} DAY => ACTUAL VALUE : {actual_value} | LSTM PREDICTED VALUE : {predictions[i]} | FINAL PREDICTION(LSTM + ARIMA) : {final_predictions[i]}\n")
    
    print(f"\n\n---------------- The FORECAST VALUE OF NEXT DATA POINT IS ------------------ \n\n")
    print(predictions[days]+predictions_errors[days])
    end1 = time.time()
    print(f"\n\nTime taken for model training and predictions: {end1 - st1:.2f} seconds\n\n")
    
    with open(os.path.join(Output_address, "output.txt"), "w+") as file:
      file.write("\n---------------- LSTM MODEL ----------------\n")
      file.write(f"The Lags Used is : {lag}\n\n")
      file.write(f"The EPOCHS is  : {epochs}\n\n")
      file.write(f"The Learning-Rate of the LSMT Model is : {learning_rate}\n\n")
      file.write(f"The Batch-Size of the LSMT Model is : {batch_size}\n\n")
      file.write(f"The Number of Nodes of the LSMT Model is  : {number_nodes}\n\n")
      file.write(f"The Lag Value for the Neural Network to Work : {n}\n\n")
      file.write("\n---------------------- FULL PREDICTIONS OF THE TRAIN DATA (FIRST 100 points) FROM LSTM MODEL -------------------------\n")
      for i in range(100):
         file.write(f"{i} => ACTUAL DATA POINT : {train.iloc[i]} | PREDICTED DATA POINT : {full_predictions[i]}\n")
      file.write(f"LMST Model Summary : \n{model.summary()}\n\n")
      file.write(f"LMST HISTORY OF THE MODEL : \n{history}\n\n")
      file.write(f"LMST Model Mean Squared Error : {mse}\n\n")
      file.write(f"LMST Model Root Mean Squared Error : {rmse}\n\n")
      file.write(f"LMST Model Mean Absolute Error : {mae}\n\n")
      file.write(f"----------------------------- THE {days} PREDICTION VALUES of LSMT MODEL -----------------------------------\n\n")
      for i, (actual, pred) in enumerate(zip(test[:days], predictions[:days])):
          file.write(f"Day {i+1} => ACTUAL VALUE: {actual} | PREDICTED VALUE: {pred}\n\n")
      file.write("\n---------------------------- ARIMA MODEL Summary -------------------------\n")
      file.write(Arima_Model.summary().as_text())
      file.write(f"\n\n---------------------------- ARIMA MODEL {days} Predictions-------------------------\n\n")
      for i in range(len(predictions_errors)):
         file.write(f"{i} : {predictions_errors[i]}\n")
      file.write("\n\n--------------------------- FINAL PREDICTIONS ---------------------------------\n\n")
      for i in range(days):
         actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
         file.write(f"\nDay {i+1} => ACTUAL VALUE : {actual_value} | PREDICTED VALUE : {final_predictions[i]}\n")
      file.write("\n---------------- Difference Between the LSTM Predictions and Final Predictions of {days} days ----------------\n")
      for i in range(days):
         actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
         file.write(f"\n{i} DAY => ACTUAL VALUE : {actual_value} | LSTM PREDICTED VALUE : {predictions[i]} | FINAL PREDICTION(LSTM + ARIMA) : {final_predictions[i]}\n")
      
      file.write(f"\nTime taken for model training and predictions: {end1 - st1:.2f} seconds\n\n")
      file.write(f"\n\n---------------- The FORECAST VALUE OF NEXT DATA POINT IS ------------------ \n\n")
      file.write(f"{predictions[days]+predictions_errors[days]}")
    print(f"Output written to {os.path.join(Output_address, 'output.txt')}")
      
    
if __name__ == '__main__':
   main()
