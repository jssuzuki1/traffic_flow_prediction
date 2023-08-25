#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import sys
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 1000)

from sklearn.model_selection import KFold, cross_val_score


# In[2]:

file_path = r"C:\Users\jssuz\OneDrive\Desktop\grad_school_files\artificial_intelligence\Automated_Traffic_Volume_Counts.csv"

# Load the CSV file into a pandas DataFrame.
orig_data = pd.read_csv(file_path)


# ## EDA

# In[3]:


## produce min/max years.
print("Minimum Year", orig_data["Yr"].min())
print("Maximum Year", orig_data["Yr"].max())

## The resulting random street was flatbush_avenue here.
# random_street = data["street"].sample().iloc[0]

# Display the randomly selected street
flatbush_avenue = orig_data[orig_data['street'] == 'FLATBUSH AVENUE']
flatbush_avenue_sorted = flatbush_avenue.sort_values(by=['Yr', 'M', 'D'], ascending=False)

# flatbush_avenue_sorted.head(100)

## Check to see if other streets exist on the same time range
street_check = orig_data[(orig_data['Yr'] == 2012) & (orig_data['M'] == 10) & (orig_data['D'] == 20)]
# print(street_check)


# ## Preprocessing Part 1: Data Transformations

# In[4]:


## Remove superfluous columns for this paper
columns_to_remove = ['RequestID', 'Boro', 'SegmentID', 'WktGeom', 'fromSt', 'toSt', 'Direction']
data = orig_data.drop(columns=columns_to_remove, axis=1)


# In[5]:


# Create the 'datetime' column
data['datetime'] = data.apply(
    lambda row: datetime(row['Yr'], row['M'], row['D'], row['HH'], row['MM']),
    axis=1
)

## Drop the former date variables
columns_to_remove = ['Yr', 'M', 'D', 'HH', 'MM']
data = data.drop(columns = columns_to_remove, axis=1)

## Check to see if filter worked:
print("Oldest Datetime", data["datetime"].min())
print("Newest Datetime", data["datetime"].max())


# In[7]:


## Checking to see if there is any data here, ensuring that those records existed for any street.
flatbush_avenue = data[data['street'] == 'FLATBUSH AVENUE']
flatbush_avenue_sorted = flatbush_avenue.sort_values(by=['datetime'], ascending=False)

flatbush_avenue_sorted


# In[8]:


# Filter datetime values between October 16, 2019, and October 31, 2019, the most recent period of continuous observation
## This is the most recent date range of the most recent observation range for Flatbush Avenue
## There is a possibility there is no missing data because there is nothing there...
start_date = datetime(2019, 10, 16)
end_date = datetime(2019, 10, 31)

data = data[
    (data['datetime'] >= start_date) &
    (data['datetime'] <= end_date)
]


# In[17]:


## Group by 'datetime' and sum the 'Vol' values
## We sum up all of the directions from which the traffic occurs to get an aggregate count of each street.
summed_data = data.groupby(['street','datetime'])['Vol'].sum().reset_index()


# In[18]:


## Check number of unique streets that exist in the current date range
unique_streets = len(data['street'].unique())

## Show the number of unique streets. 
## This is a small fraction of the original 6,750 number of streets.
print('No. of unique streets between October 16, 2019, and October 31, 2023:', unique_streets)


# In[19]:


## Check to see if filter worked:
print("Oldest Datetime after filter and sum", summed_data["datetime"].min())
print("Newest Datetime after filter and sum", summed_data["datetime"].max())


# In[220]:


summed_data


# In[20]:


# Pivot the DataFrame to create one street per column.
pivoted_data = summed_data.pivot_table(index='datetime', columns='street', values='Vol')

pivoted_data = pivoted_data.reset_index()


# In[24]:


## This actually returns that there is an uninterrupted period of observation for flatbush between October 16, 2019, and October 31, 2023.
## This should be empty.
missing_values = pivoted_data[pd.isna(pivoted_data['FLATBUSH AVENUE'])][['FLATBUSH AVENUE', 'datetime']]
print('Missing values for flatbush avenue (should be 0):', len(missing_values))


# In[47]:


## lead_flatbush will serve as our predicted variable.
## This is a leading variable-- the actual value 5 minutes in the future.
target_variable = pivoted_data['FLATBUSH AVENUE'].shift(-1).iloc[:-1]

## Remove the very last record, since the final record cannot predict anything.
preprocessed_data = pivoted_data.drop(pivoted_data.index[-1])

## Creaete a time variable to serve as a predictor-- the time of the day factors into traffic.
# Convert 'datetime' column to datetime format
preprocessed_data['datetime'] = pd.to_datetime(preprocessed_data['datetime'])
                                                                                              
# Extract time component and convert to seconds since midnight
preprocessed_data['time'] = preprocessed_data['datetime'].dt.hour * 3600 + preprocessed_data['datetime'].dt.minute * 60 + preprocessed_data['datetime'].dt.second

# Remove 'datetime' variable
preprocessed_data = preprocessed_data.drop(columns=['datetime'])


# In[51]:


## Check the rows with NaN values
# missing_value_counts = preprocessed_data.isna().sum()
# print(missing_value_counts)


# In[58]:


## Drop all columns with NA values.
preprocessed_data = preprocessed_data.dropna(axis=1)

## Testing with a far smaller data set for now prior to production:
preprocessed_data = preprocessed_data
target_variable = target_variable

## When we get rid of all of the entries with missing values, we get 18 streets
print('Shape of Preprocessed Data:', preprocessed_data.shape)
print('input data length:', len(preprocessed_data))
print('target variable length:', len(target_variable))


# ## Preprocessing Part 2: Sequences, Batches, and Test-Train Splits

# In[59]:


# Define the number of timesteps to use for prediction
num_timesteps = 8

# Convert input_data DataFrame to a NumPy array
input_array = preprocessed_data.values

# Define the number of timesteps to use for prediction
num_timesteps = 8

# Create sequences of 8 timesteps each for prediction
sequences = []
target_values = []

for i in range(len(input_array) - num_timesteps):
    sequence = input_array[i : i + num_timesteps]
    sequences.append(sequence)


# In[60]:


target_variable = target_variable[8:]  # Start from the 9th entry onwards


# In[61]:


print('target length:', len(target_variable))
print('input data length:', len(sequences))


# In[62]:


## Split into Test-Train Set

total_samples = len(sequences)
train_samples = int(total_samples * 0.8)  # 80% of the total samples

input_train = sequences[:train_samples]
input_test = sequences[train_samples:]

target_train = target_variable[:train_samples]
target_test = target_variable[train_samples:]


# In[63]:


print('TRAIN input length:', len(input_train))
print('TEST input length:', len(input_test))

print('TRAIN target length:', len(target_train))
print('TEST target length:', len(target_test))

## Remove the bottom two entries to target_variable and sequences.
round_by_ten = len(input_train) // 10 * 10
print('Rounded training set size:', round_by_ten)


# In[64]:


## BATCHES

## Remove the "ones" place of the records
input_train = input_train[:round_by_ten]
target_train = target_train[:round_by_ten]

# Convert lists to Numpy arrays
sequences = np.array(input_train)
target_variable = np.array(target_train)
num_features = preprocessed_data.shape[1]

# Reshape input_data and target_variable to match (num_samples, timesteps, num_features) format
sequences_reshaped = sequences.reshape(sequences.shape[0], sequences.shape[1], num_features)
target_variable_reshaped = target_variable.reshape(target_variable.shape[0], 1)  # Reshape target_variable to add a dimension

# Apply batching on reshaped input_data and target_variable
batch_size = 10
num_samples, timesteps, num_features = sequences_reshaped.shape
num_batches = num_samples // batch_size

# # Remove the extra batch dimension from sequences_reshaped
# sequences_reshaped = sequences_reshaped[:num_batches * batch_size].reshape(num_batches, batch_size, timesteps, num_features)
# target_variable_batched = target_variable_reshaped[:num_batches * batch_size].reshape(num_batches, batch_size, 1)

# Calculate the number of samples that can be evenly divided into batches
samples_per_batch = num_batches * batch_size

# Apply batching on reshaped input_data and target_variable
sequences_reshaped = sequences_reshaped[:samples_per_batch].reshape(num_batches, batch_size, timesteps, num_features)
target_variable_batched = target_variable_reshaped[:samples_per_batch].reshape(num_batches, batch_size, 1)

# Sample target values corresponding to each input sequence
## In my actual code, this would just would just be the predicted vector.
target_values = [tf.constant(target_variable, dtype=tf.float32) for _ in data]


# ## GRU

# In[206]:


# Define hyperparameters
hidden_size = 19
output_size = 1

# Define optimizer and loss function
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Clip gradients by value

## Loss here is just meanSquaredError.
loss_fn = tf.keras.losses.MeanSquaredError()

class GRUNet(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        ## super() calls a method from the parent class.
        ## this ensures the initialization of the parent class.
        super(GRUNet, self).__init__()
        
        ## Creates an instance of a GRU layer.
        self.gru = tf.keras.layers.GRU(hidden_size, return_sequences=True)
        
        ## these are intermittent layers.
        ## Too few neurons would be underfitting, while too many neurons would be overfitting.
        self.fc1 = tf.keras.layers.Dense(10, activation='relu')

        ## Creates a fully connected dense layer ("fully connected"). The final layer will always be the size of the desired output.
        self.final = tf.keras.layers.Dense(output_size)
        
    ## Defines the forward pass of my model.
    ## The three 'outs' overwrite the contents of the previous 'out'
    def call(self, x):
        out = self.gru(x)
        out = self.fc1(out[:, -1, :])
        out = self.final(out)
        return out
    
# Create an instance of the extended LSTNet class
gru_net = GRUNet(hidden_size, output_size)


# In[207]:


num_epochs = 100
max_iterations = 114 ## The number of batches that exist.
latest_epoch = num_epochs - 1  # Index of the latest epoch

for epoch in range(num_epochs):
    epoch_loss = 0.0
    latest_epoch_predicted_outputs = []  # List to store predicted outputs for the latest epoch

    for iteration in range(max_iterations):  # Change the loop to iterate over max_iterations
        batch_sequences = sequences_reshaped[iteration % num_batches]  # Modulo operation for cycling through batches
        batch_target_values = target_variable_batched[iteration % num_batches]

        with tf.GradientTape() as tape:
            batch_outputs = gru_net(batch_sequences)
            loss = loss_fn(y_true=batch_target_values, y_pred=batch_outputs)
        gradients = tape.gradient(loss, gru_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gru_net.trainable_variables))
        epoch_loss += loss.numpy()

        # Store predicted output only for the latest epoch
        if epoch == latest_epoch:
            latest_epoch_predicted_outputs.append(batch_outputs.numpy())

    print(f"Epoch {epoch + 1}, Loss: {np.sqrt(epoch_loss / max_iterations):.4f}")

# Convert the list of predicted outputs for the latest epoch into a NumPy array
GRU_predicted_outputs = np.concatenate(latest_epoch_predicted_outputs, axis=0)


# In[84]:


# print(len(np.array(latest_epoch_predicted_outputs)))

# for element in latest_epoch_predicted_outputs:
#     print(element)


# ## LSTM

# In[208]:


# Define hyperparameters for BOTH GRU and LSTM
hidden_size = 19
output_size = 1

# Define optimizer and loss function
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Clip gradients by value

## Loss here is just meanSquaredError.
loss_fn = tf.keras.losses.MeanSquaredError()

class LSTMNet(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        ## super() calls a method from the parent class.
        ## this ensures the initialization of the parent class.
        super(LSTMNet, self).__init__()
        
        ## Creates an instance of a GRU layer.
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        
        self.fc1 = tf.keras.layers.Dense(10, activation='relu')

        ## Creates a fully connected dense layer ("fully connected"). The final layer will always be the size of the desired output.
        self.final = tf.keras.layers.Dense(output_size)
        
    ## Defines the forward pass of my model.
    ## The three 'outs' overwrite the contents of the previous 'out'
    def call(self, x):
        out = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.final(out)
        return out
    
# Create an instance of the extended LSTNet class
lstm_net = LSTMNet(hidden_size, output_size)


# In[209]:


num_epochs = 100
max_iterations = 114 ## The number of batches that exist.
latest_epoch = num_epochs - 1  # Index of the latest epoch

for epoch in range(num_epochs):
    epoch_loss = 0.0
    latest_epoch_predicted_outputs = []  # List to store predicted outputs for the latest epoch

    for iteration in range(max_iterations):  # Change the loop to iterate over max_iterations
        batch_sequences = sequences_reshaped[iteration % num_batches]  # Modulo operation for cycling through batches
        batch_target_values = target_variable_batched[iteration % num_batches]

        with tf.GradientTape() as tape:
            batch_outputs = lstm_net(batch_sequences)
            loss = loss_fn(y_true=batch_target_values, y_pred=batch_outputs)
        gradients = tape.gradient(loss, lstm_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, lstm_net.trainable_variables))
        epoch_loss += loss.numpy()

        # Store predicted output only for the latest epoch
        if epoch == latest_epoch:
            latest_epoch_predicted_outputs.append(batch_outputs.numpy())

    print(f"Epoch {epoch + 1}, Loss: {np.sqrt(epoch_loss / max_iterations):.4f}")

# Convert the list of predicted outputs for the latest epoch into a NumPy array
LSTM_predicted_outputs = np.concatenate(latest_epoch_predicted_outputs, axis=0)


# In[181]:


len(LSTM_latest_epoch_predicted_outputs)


# ## Apply To Models to Test

# In[210]:


# gru_net.summary()


# In[211]:


test_loss = 0.0
test_predictions = []  # List to store predicted outputs for the test set

for seq, target in zip(input_test, target_test):
    output = gru_net(seq[tf.newaxis, ...])
    test_loss += loss_fn(y_true=target, y_pred=output).numpy()

    # Store the predicted output for the test set
    test_predictions.append(output.numpy())

print(f"Test Loss: {np.sqrt(test_loss / len(input_test)):.4f}")

# Convert the list of predicted outputs for the test set into a NumPy array
gru_test_predictions = np.concatenate(test_predictions, axis=0)


# In[212]:


# len(gru_test_predictions)


# In[193]:


# lstm_net.summary()


# In[213]:


test_loss = 0.0
test_predictions = []  # List to store predicted outputs for the test set

for seq, target in zip(input_test, target_test):
    output = lstm_net(seq[tf.newaxis, ...])
    test_loss += loss_fn(y_true=target, y_pred=output).numpy()

    # Store the predicted output for the test set
    test_predictions.append(output.numpy())

print(f"Test Loss: {np.sqrt(test_loss / len(input_test)):.4f}")

# Convert the list of predicted outputs for the test set into a NumPy array
lstm_test_predictions = np.concatenate(test_predictions, axis=0)


# In[188]:


# len(lstm_test_predictions)


# In[572]:


# test_predictions = []  # List to store predicted outputs for the test set

# for iteration in range(max_iterations):
#     batch_sequences = input_test[iteration % num_batches]  # Modulo operation for cycling through batches

#     # Predict for the batch using the trained model
#     batch_outputs = gru_net(batch_sequences)

#     # Store the predicted outputs for the test set
#     test_predictions.append(batch_outputs.numpy())

# # Convert the list of predicted outputs for the test set into a NumPy array
# test_predictions = np.concatenate(test_predictions, axis=0)


# ## Graphics - Charts for GRU and LSTM

# In[113]:


print(len(gru_test_predictions))
print(len(target_test))


# In[214]:


## Merging GRU Data for a graph. 

# Convert the numpy arrays to Pandas Series
target_series = pd.Series(target_test, name='Actual').reset_index(drop=True)
predicted_series = pd.Series(gru_test_predictions[:, 0], name='Predicted').reset_index(drop=True)

# Create a DataFrame by combining the Series
GRU_act_v_predicted = pd.concat([target_series, predicted_series], axis=1)

## Add loss column:
GRU_act_v_predicted['Loss'] = GRU_act_v_predicted['Actual'] - GRU_act_v_predicted['Predicted'] 

# Print the resulting DataFrame
GRU_act_v_predicted


# In[215]:


## Merging LSTM Data for a graph. 

# Convert the numpy arrays to Pandas Series
target_series = pd.Series(target_test, name='Actual').reset_index(drop=True)
predicted_series = pd.Series(lstm_test_predictions[:, 0], name='Predicted').reset_index(drop=True)

# Create a DataFrame by combining the Series
LSTM_act_v_predicted = pd.concat([target_series, predicted_series], axis=1)

## Add loss column:
LSTM_act_v_predicted['Loss'] = LSTM_act_v_predicted['Actual'] - LSTM_act_v_predicted['Predicted'] 

# Print the resulting DataFrame
LSTM_act_v_predicted


# In[216]:


# Create a DataFrame with increments of 200 records
increment = 200

# Set up Seaborn style
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=GRU_act_v_predicted[["Actual", "Predicted"]])  # Select only the "Actual" and "Predicted" columns

# Set labels and title
plt.xlabel("Time Step")
plt.ylabel("Traffic Flow Volume")
plt.title("Actual vs Predicted - GRU")

# Show the plot
plt.show()


# In[217]:


# Create a DataFrame with increments of 200 records
increment = 200

# Set up Seaborn style
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data= LSTM_act_v_predicted[["Actual", "Predicted"]])  # Select only the "Actual" and "Predicted" columns

# Set labels and title
plt.xlabel("Time Step")
plt.ylabel("Traffic Flow Volume")
plt.title("Actual vs Predicted - LSTM")

# Show the plot
plt.show()


# ## Loss Over Time

# In[218]:


# Create a DataFrame with increments of 200 records
increment = 200

# Set up Seaborn style
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=GRU_act_v_predicted["Loss"])  # Select only the "Actual" and "Predicted" columns

# Set labels and title
plt.xlabel("Time Step")
plt.ylabel("Loss")
plt.title("Loss Over Time- GRU")

# Show the plot
plt.show()


# In[219]:


# Create a DataFrame with increments of 200 records
increment = 200

# Set up Seaborn style
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=LSTM_act_v_predicted["Loss"])  # Select only the "Actual" and "Predicted" columns

# Set labels and title
plt.xlabel("Time Step")
plt.ylabel("Loss")
plt.title("Loss Over Time- GRU")

# Show the plot
plt.show()

