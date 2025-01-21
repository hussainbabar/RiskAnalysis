import streamlit as st 
import pickle 
import numpy as np 

# load the saved model 

model = pickle.load(open("simpleLinear.pkl",'rb'))

#set the title 

st.title("Area price Prediction")

# Add the brief description

st.write("This application predicts based on the Square feet living value")

# Add input 

sqft_area = st.number_input("Enter your Sqft:", min_value=0.0, max_value=10000.0, value=1.0, step=0.5)

# When the button is clicked, make predictions

if st.button("Predict Area price"):
    area_input = np.array([[sqft_area]])
    prediction = model.predict(area_input)

# display output
st.success("The predcited area of {sqft_area} price of the sqft is ${prediction[0]:,.2f}")