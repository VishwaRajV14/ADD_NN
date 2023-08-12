
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

# TEST THE MODEL
model = tf.keras.models.load_model(r'C:\Users\Vishwa Raj\OneDrive\Desktop\ADD_NN\sum_model.model')
op1 = input("Enter the first operand:")
op2 = input("Enter the second operand:")
test_input = np.array([[op1,op2]])
sum = model.predict(test_input)
print("Sum is :",sum)
