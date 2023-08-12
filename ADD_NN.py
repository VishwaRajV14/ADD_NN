import pandas as pd
import numpy as np
import tensorflow as tf

excel_path = r'C:\Users\Vishwa Raj\OneDrive\Desktop\ADD_NN\Addition_data_set.xlsx'
df = pd.read_excel(excel_path)
ip_column_names = ['Operand1','Operand2']
op_column_names = ['Result']
input_train_data = df[ip_column_names].values
output_train_data = df[op_column_names].values
output_train_data = np.array(output_train_data).reshape(-1,1)

#input_train_data = np.array[input_train_data]
#output_train_data = np.array[output_train_data]

# Create the model now 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(15,activation='relu',input_shape = (2,)))
model.add(tf.keras.layers.Dense(30,activation='relu'))
model.add(tf.keras.layers.Dense(30,activation='relu'))
model.add(tf.keras.layers.Dense(15,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='linear'))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(input_train_data,output_train_data,epochs=200,batch_size=50)
model.save(r'C:\Users\Vishwa Raj\OneDrive\Desktop\ADD_NN\sum_model.model')
