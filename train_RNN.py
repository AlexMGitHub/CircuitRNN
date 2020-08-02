#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:19:13 2020

@author: alex

###############################################################################
# train_RNN.py
#
# Revision:     1.00
# Date:         8/1/2020
# Author:       Alex
#
# Purpose:      Generates time series data using the Circuitry module and trains
#               an LSTM and GRU on the data to compare their performance.
#
# Inputs:
# 1. Circuit parameters to generate_data() function
# 2. plot_pts, number of points to plot
#
# Outputs:
# 1. Training time of LSTM and GRU
# 2. Three plots of RNN performance on training data, test data, and 
#    forecast test data
# 3. Mean squared error of the RNNs on training data, test data, and forecasted
#    test data
# 4. An animated plot of the forecasted test data.
#
# Notes:
# 1. Code based on LSTM training example at the following URL:
#    https://apmonitor.com/do/index.php/Main/LSTMNetwork
#
##############################################################################
"""
#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import Circuitry as circuit

# Keras imports for LSTM and GRU models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.models import load_model


#%% Functions
def generate_data(nbits=8, tau=1, seed=1, noise_std = 0.2, data_len=2**16,
                  plot_pts=100):
    dac8 = circuit.DAC(nbits=nbits)
    sr = circuit.ShiftRegister(nbits=nbits)
    rc = circuit.RC_Circuit(tau=tau)
    input_list = []
    dac_out_list = []
    noisy_out_list = []
    voltage_list = []
    rand_gen = np.random
    rand_gen.seed(seed)
    for i in range(data_len):
        input_bit = rand_gen.randint(0,2)
        dac_output = dac8.clock_inputs(sr.clock_inputs(input_bit))
        noisy_output = rand_gen.normal(dac_output, scale=noise_std)
        voltage = rc.apply_voltage(noisy_output, tdelta=1)
        input_list.append(input_bit)
        dac_out_list.append(dac_output)
        noisy_out_list.append(noisy_output)
        voltage_list.append(voltage)
    # Plot data
    xaxis = range(1,plot_pts+1)
    plt.figure(figsize=(12,9))
    ax1 = plt.subplot(2,1,1)
    plt.plot(xaxis,dac_out_list[0:plot_pts],'r-',label='DAC Output')
    plt.plot(xaxis,noisy_out_list[0:plot_pts],'g:',label='Noisy Output')
    plt.plot(xaxis,voltage_list[0:plot_pts],'k--',label='RC Output')
    plt.legend()
    plt.ylabel('Voltage (V)')
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.plot(xaxis,input_list[0:plot_pts],'b-',label='Input bits')
    plt.xlabel('Clock cycle'); plt.ylabel('Input bits')
    plt.legend()
    plt.savefig('docs/img/Generated_Data.png')
    plt.show()
    return input_list, voltage_list


#%% Generate data and split into test/train sets
plot_pts = 100 # Num points to plot
input_list, voltage_list = generate_data(nbits=8, tau=1, seed=1, noise_std = 0.2,
                             data_len=2**16, plot_pts=plot_pts)

# Split data into test/train sets without shuffling
Xtrain, Xtest, ytrain, ytest = train_test_split(input_list, voltage_list, 
                                                shuffle=False, train_size=0.85)

# Scale input features
mms_input = MinMaxScaler(feature_range=(-1,1))
Xtrain = mms_input.fit_transform(np.column_stack([Xtrain,ytrain]))
Xtest = mms_input.transform(np.column_stack([Xtest,ytest]))

# Scale output values
mms_output = MinMaxScaler(feature_range=(-1,1))
ytrain = mms_output.fit_transform(np.array(ytrain).reshape(-1,1))
ytest = mms_output.transform(np.array(ytest).reshape(-1,1))

# Group training data into sub-arrays using a sliding window
window = 8 # 8-bit shift register and DAC
Xtrain_window = []
ytrain_window = []
for i in range(window,len(Xtrain)):
    Xtrain_window.append(Xtrain[i-window:i,:])
    ytrain_window.append(ytrain[i])
Xtrain_window, ytrain_window = np.array(Xtrain_window), np.array(ytrain_window)


#%% Create and train LSTM model
num_epochs = 20
# Initialize LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, \
          input_shape=(Xtrain_window.shape[1],Xtrain_window.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit (and time) LSTM model
t0 = time.time()
history = model.fit(Xtrain_window, ytrain_window, epochs=num_epochs, 
                    batch_size=250, verbose=1)
t1 = time.time()
lstm_runtime = t1-t0
print('LSTM Runtime: %.2f s' %(lstm_runtime))
model.save('models/lstm_model.h5')
lstm_loss = history.history['loss']


#%% Create and train GRU model
# Initialize GRU model
model = Sequential()
model.add(GRU(units=50, return_sequences=True, \
          input_shape=(Xtrain_window.shape[1],Xtrain_window.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit (and time) GRU model
t0 = time.time()
history = model.fit(Xtrain_window, ytrain_window, epochs=num_epochs, 
                    batch_size=250, verbose=1)
t1 = time.time()
gru_runtime = t1-t0
print('GRU Runtime: %.2f s' %(gru_runtime))
model.save('models/gru_model.h5')
gru_loss = history.history['loss']

# Plot training loss
plt.figure(figsize=(12,9))
plt.semilogy(range(1,num_epochs+1), lstm_loss,'-r',label='LSTM Loss')
plt.semilogy(range(1,num_epochs+1), gru_loss,'-g',label='GRU Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.xlim(1,num_epochs)
plt.grid()
plt.legend()
plt.savefig('docs/img/Training_Loss.png')


#%% Compare GRU and LSTM prediction performance on training data
# Load models
lstm_model = load_model('models/lstm_model.h5')
gru_model = load_model('models/gru_model.h5')

# Verify the fit of the models on training data
ytrain_lstm = lstm_model.predict(Xtrain_window)
ytrain_gru = gru_model.predict(Xtrain_window)

# Unscale training inputs and outputs
Xtrain_unscaled = mms_input.inverse_transform(Xtrain)
ytrain_lstm = mms_output.inverse_transform(ytrain_lstm)
ytrain_gru = mms_output.inverse_transform(ytrain_gru)
ytrain_unscaled = mms_output.inverse_transform(ytrain_window)

# Plot prediction on training data versus ground truth
xaxis = range(window,plot_pts+window)
plt.figure(figsize=(12,9))
ax1 = plt.subplot(2,1,1)
plt.plot(xaxis,ytrain_lstm[0:plot_pts],'r-',label='LSTM Prediction on Training Data')
plt.plot(xaxis,ytrain_gru[0:plot_pts],'g-',label='GRU Prediction on Training Data')
plt.plot(xaxis,ytrain_unscaled[0:plot_pts],'k--',label='Training Ground Truth')
plt.ylabel('Voltage (V)')
plt.legend()
ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(xaxis,Xtrain_unscaled[window:plot_pts+window,0],'b-',label='Input bits')
plt.legend()
plt.xlabel('Clock cycle'); plt.ylabel('Input bits')
plt.savefig('docs/img/Training_Data.png')
plt.show()

# Calculate MSE
mse_train_lstm = np.sum((ytrain_lstm - ytrain_unscaled)**2)/len(ytrain_unscaled)
mse_train_gru = np.sum((ytrain_gru - ytrain_unscaled)**2)/len(ytrain_unscaled)
print('The MSE of LSTM predictions on training data = {}'.format(mse_train_lstm))
print('The MSE of GRU predictions on training data = {}'.format(mse_train_gru))


#%% Compare GRU and LSTM prediction performance on test data
# Group test data into sub-arrays using a sliding window
Xtest_window = []
ytest_window = []
for i in range(window,len(Xtest)):
    Xtest_window.append(Xtest[i-window:i,:])
    ytest_window.append(ytest[i])
Xtest_window, ytest_window = np.array(Xtest_window), np.array(ytest_window)

# Predict on test data
ytest_lstm = lstm_model.predict(Xtest_window)
ytest_gru = gru_model.predict(Xtest_window)

# Unscale test inputs and outputs
Xtest_unscaled = mms_input.inverse_transform(Xtest)
ytest_lstm = mms_output.inverse_transform(ytest_lstm)
ytest_gru = mms_output.inverse_transform(ytest_gru)
ytest_unscaled = mms_output.inverse_transform(ytest_window)

# Plot results of test data
xaxis = range(window,plot_pts+window)
plt.figure(figsize=(12,9))
ax1 = plt.subplot(2,1,1)
plt.plot(xaxis,ytest_lstm[0:plot_pts],'r-',label='LSTM Prediction on Test Data')
plt.plot(xaxis,ytest_gru[0:plot_pts],'g-',label='GRU Prediction on Test Data')
plt.plot(xaxis,ytest_unscaled[0:plot_pts],'k--',label='Test Ground Truth')
plt.legend()
plt.ylabel('Voltage (V)')
ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(xaxis,Xtest_unscaled[window:plot_pts+window,0],
         'b-',label='Input bits')
plt.xlabel('Clock cycle'); plt.ylabel('Input bits')
plt.legend()
plt.savefig('docs/img/Test_Data.png')
plt.show()

# Calculate MSE
mse_test_lstm = np.sum((ytest_lstm - ytest_unscaled)**2)/len(ytest_unscaled)
mse_test_gru = np.sum((ytest_gru - ytest_unscaled)**2)/len(ytest_unscaled)
print('The MSE of LSTM predictions on test data = {}'.format(mse_test_lstm))
print('The MSE of GRU predictions on test data = {}'.format(mse_test_gru))


#%% Compare GRU and LSTM forecasting performance on test data
# Using predicted values to predict next step
Xforecast_lstm = Xtest.copy()
yforecast_lstm = ytest_window.copy()
Xforecast_gru = Xtest.copy()
yforecast_gru = ytest_window.copy()
for i in range(window,len(Xforecast_lstm)):
    Xin = Xforecast_lstm[i-window:i].reshape((1, window, 2))
    Xforecast_lstm[i][1] = lstm_model.predict(Xin)
    yforecast_lstm[i-window] = Xforecast_lstm[i][1]
    Xin = Xforecast_gru[i-window:i].reshape((1, window, 2))
    Xforecast_gru[i][1] = gru_model.predict(Xin)
    yforecast_gru[i-window] = Xforecast_gru[i][1]

# Unscale forecast output
yforecast_lstm = mms_output.inverse_transform(yforecast_lstm) 
yforecast_gru = mms_output.inverse_transform(yforecast_gru) 

# Plot results of the forecast
xaxis = range(len(Xtest_unscaled)-plot_pts+1, len(Xtest_unscaled)+1)
plt.figure(figsize=(12,9))
ax1 = plt.subplot(2,1,1)
plt.plot(xaxis,yforecast_lstm[-plot_pts:],'r-',label='LSTM Forecast')
plt.plot(xaxis,yforecast_gru[-plot_pts:],'g-',label='GRU Forecast')
plt.plot(xaxis,ytest_unscaled[-plot_pts:],'k--',label='Ground Truth')
plt.legend()
plt.ylabel('Voltage (V)')
ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(xaxis,Xtest_unscaled[-plot_pts:,0],'b-',label='Input bits')
plt.xlabel('Clock cycle'); plt.ylabel('Input bits')
plt.legend()
plt.savefig('docs/img/Forecast.png')
plt.show()

# Calculate MSE
mse_forecast_lstm = np.sum((yforecast_lstm - ytest_unscaled)**2)/len(ytest_unscaled)
mse_forecast_gru = np.sum((yforecast_gru - ytest_unscaled)**2)/len(ytest_unscaled)
print('The MSE of the LSTM forecast on test data = {}'.format(mse_forecast_lstm))
print('The MSE of the GRU forecast on test data = {}'.format(mse_forecast_gru))


#%% Animate plot
import matplotlib.animation as animation

# Plot results of the forecast
xaxis = range(len(Xtest_unscaled)-plot_pts+1, len(Xtest_unscaled)+1)
fig=plt.figure(figsize=(12,9))
ax1 = plt.subplot(2,1,1)
line1,=plt.plot(xaxis,yforecast_lstm[-plot_pts:],'r-',label='LSTM Forecast')
line2,=plt.plot(xaxis,yforecast_gru[-plot_pts:],'g-',label='GRU Forecast')
line3,=plt.plot(xaxis,ytest_unscaled[-plot_pts:],'k--',label='Ground Truth')
plt.legend()
plt.ylabel('Voltage (V)')
ax2 = plt.subplot(2,1,2, sharex=ax1)
line4,=plt.plot(xaxis,Xtest_unscaled[-plot_pts:,0],'b-',label='Input bits')
plt.xlabel('Clock cycle'); plt.ylabel('Input bits')
plt.legend()

x = xaxis
y = yforecast_lstm[-plot_pts:]
z = yforecast_gru[-plot_pts:]
r = ytest_unscaled[-plot_pts:]
q = Xtest_unscaled[-plot_pts:,0]

def update(num, x, y, z, r, q, line1, line2, line3, line4):
    line1.set_data(x[:num], y[:num])
    line2.set_data(x[:num], z[:num])
    line3.set_data(x[:num], r[:num])
    line4.set_data(x[:num], q[:num])
    return [line1,line2,line3,line4]

ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, z, r, q, line1, 
                              line2, line3, line4], interval=100, blit=True)

writer = animation.FFMpegWriter(
    fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("docs/img/Forecast.mp4", writer=writer)

plt.show()