from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Flatten, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf

def MLP_model(input_shape):
    inputs = Input(shape=input_shape, name="input")

    # Branch 1
    x1 = Dense(128, activation="relu", name="dense_1_branch1")(inputs)
    x1 = Dense(128, activation="relu", name="dense_2_branch1")(x1)
    x1 = Dense(64, activation="relu", name="dense_3_branch1")(x1)

    # Branch 2
    x2 = Dense(128, activation="relu", name="dense_1_branch2")(inputs)
    x2 = Dense(128, activation="relu", name="dense_2_branch2")(x2)
    x2 = Dense(64, activation="relu", name="dense_3_branch2")(x2)

    # Branch 3
    x3 = Dense(128, activation="relu", name="dense_1_branch3")(inputs)
    x3 = Dense(128, activation="relu", name="dense_2_branch3")(x3)
    x3 = Dense(64, activation="relu", name="dense_3_branch3")(x3)

    # Branch 4
    x4 = Dense(128, activation="relu", name="dense_1_branch4")(inputs)
    x4 = Dense(128, activation="relu", name="dense_2_branch4")(x4)
    x4 = Dense(64, activation="relu", name="dense_3_branch4")(x4)

    # Combine branches
    combined = Concatenate(name="concat")([x1, x2, x3, x4])

    # Fully connected layers
    x = Dense(256, activation="relu", name="dense_1")(combined)
    x = Dense(256, activation="relu", name="dense_2")(x)
    outputs = Dense(3, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def CNN_model(input_shape):
    inputs = Input(shape=input_shape, name="input")

    x = Conv2D(32, (1, 1), activation="relu", padding="same", name="conv1")(inputs)
    x = Conv2D(64, (1, 1), activation="relu", padding="same", name="conv2")(x)
    x = Conv2D(128, (1, 1), activation="relu", padding="same", name="conv3")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(128, activation="relu", name="dense_1")(x)
    outputs = Dense(3, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def RNN_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape, name = "input")

    # RNN Layer
    x = LSTM(128, return_sequences=True, name = "lstm_1")(inputs)
    x = LSTM(64, name = "lstm_2")(x)

    # Fully connected layers
    x = Dense(128, activation="relu", name = "dense_1")(x)
    outputs = Dense(3, activation="softmax", name = "output")(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

import streamlit as st
import numpy as np
import tensorflow as tf

# Define the model architecture
input_shape = (18,)  # Adjust input shape as per your model
mlp_model = MLP_model(input_shape)

input_shape_cnn = (18, 1, 1)
cnn_model = CNN_model(input_shape_cnn)

input_shape_rnn = (18, 1)
rnn_model = RNN_model(input_shape_rnn)

# Load pre-trained model weights
try:
    mlp_model.load_weights("Model_weights\Model_selection\model_mlp.h5")
    cnn_model.load_weights("Model_weights\Model_selection\model_cnn.h5")
    rnn_model.load_weights("Model_weights\Model_selection\model_rnn.h5")
    st.success("Model weights loaded successfully.")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Set the title of the app
st.title("Diabetes Prediction UI")

# User Inputs
st.header("Enter Your Health Information")

# Input fields corresponding to the dataset columns
high_bp = st.number_input("Enter your level of high blood pressure (0 = None, 1 = Very High)", min_value=0.0, max_value=1.0, value=0.5)
high_chol = st.number_input("Enter your level of high cholesterol (0 = None, 1 = Very High)", min_value=0.0, max_value=1.0, value=0.5)
chol_check = st.selectbox("Have you undergone a cholesterol check? (0 = No, 1 = Yes)", [0, 1])
height = st.number_input("Enter your height (in cm)", min_value=10.0, max_value=200.0, value=100.0)
weight = st.number_input("Enter your weight (in kg)", min_value=0.0, max_value=120.0, value=30.0)
bmi = (weight/height ** 2) * 10000
smoker = st.selectbox("Enter your smoking frequency (0 = Never, 1 = Heavy Smoker", [0, 1])
stroke = st.selectbox("Have you ever get stroke? (0 = None, 1 = Very High)", [0, 1])
heart_disease = st.selectbox("Have you ever get any heart disease or attack (0 = None, 1 = Very High))", [0, 1])
phys_activity = st.selectbox("Enter your level of physical activity (0 = None, 1 = Very Active)", [0, 1])
fruits = st.selectbox("Enter your frequency of fruit consumption (0 = Never, 1 = Daily)", [0, 1])
veggies = st.selectbox("Enter your frequency of vegetable consumption (0 = Never, 1 = Daily)", [0, 1])
hvy_alcohol = st.selectbox("Enter your level of heavy alcohol consumption (0 = None, 1 = Very High)", [0, 1])
no_doc_cost = st.selectbox("Enter how often you avoid seeing a doctor due to cost (0 = Never, 1 = Always)", [0, 1])
gen_hlth = st.slider("How would you rate your general health? (1 = Poor, 5 = Excellent)", min_value=1, max_value=5, value=2)
ment_hlth = st.slider("How many number of mentally unhealthy days you experience in a month?", min_value=1, max_value=30, value=5)
phys_hlth = st.slider("How many number of physically unhealthy days you experience in a month?", min_value=1, max_value=30, value=5)
diff_walk = st.selectbox("Enter your difficulty walking (0 = None, 1 = Severe)", [0, 1])
sex = st.selectbox("Select your gender", ["Female", "Male"])
if sex == "Female":
    sex = 0
else:
    sex = 1
age = st.slider("Please specify your age", min_value=1, max_value=100, value=18)

# Model Selection
st.header("Select a Deep Learning Model")
model_choice = st.radio("Choose a model", ("MLP", "CNN", "RNN"))

# Convert inputs to numpy array and make a prediction
if st.button("Run Model"):
    # Prepare the input data for prediction
    input_data = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity,
                            fruits, veggies, hvy_alcohol, no_doc_cost, gen_hlth, ment_hlth,
                            phys_hlth, diff_walk, sex, age//5]])
    
    try:
        # Make a prediction using the model
        if model_choice == "MLP":
            prediction = mlp_model.predict(input_data)
        elif model_choice == "CNN":
            input_data = input_data.reshape(-1, input_data.shape[1], 1, 1)
            prediction = cnn_model.predict(input_data)
        elif model_choice == "RNN":
            input_data = input_data.reshape(-1, input_data.shape[1], 1)
            prediction = rnn_model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        labels = ["No Diabetes", "Prediabetes", "Diabetes"]  # Adjust as per your classes
        prediction_label = labels[predicted_class]
        
        # Display the result
        st.write(f"The model predicts: {prediction_label}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
