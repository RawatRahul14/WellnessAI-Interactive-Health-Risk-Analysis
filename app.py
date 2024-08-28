import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model

# Define the Keras models as you already have
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

def MLP_2_Model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation="relu", name="layer_1")(inputs)
    x = BatchNormalization(name = "layer_1_batchnorm1")(x)
    x = Dropout(0.4, name = "layer_1_drop1")(x)
    x = Dense(256, activation="relu", name="layer_2")(x)
    x = BatchNormalization(name = "layer_2_batchnorm1")(x)
    x = Dropout(0.4, name = "layer_2_drop1")(x)
    x = Dense(256, activation="relu", name="layer_3")(x)
    x = BatchNormalization(name = "layer_3_batchnorm1")(x)
    x = Dropout(0.4, name = "layer_3_drop1")(x)
    outputs = Dense(3, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Load the pre-trained Keras model weights
input_shape = (18,)
mlp_model = MLP_model(input_shape)
mlp_2_model = MLP_2_Model(input_shape)

try:
    mlp_model.load_weights("Model_weights\Model_selection\model_mlp.h5")
    mlp_2_model.load_weights("Model_weights\Model_selection\model_2_mlp.h5")
    st.success("Keras model weights loaded successfully.")
except Exception as e:
    st.error(f"Error loading Keras model weights: {e}")
    st.stop()

# Load the pre-trained Random Forest model
try:
    rf_model = joblib.load("Model_weights\Model_selection\model_rf.pkl")
    st.success("Random Forest model loaded successfully.")
except Exception as e:
    st.error(f"Error loading Random Forest model: {e}")
    st.stop()

# Set the title of the app
st.title("Diabetes Prediction UI")

# User Inputs
st.header("Enter Your Health Information")
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
def gen_hlth_exp(gen_hlth):
    if gen_hlth == 1:
        return 5
    elif gen_hlth == 2:
        return 4
    elif gen_hlth == 3:
        return 3
    elif gen_hlth == 4:
        return 2
    elif gen_hlth == 5:
        return 1
gen_hlth = gen_hlth_exp(gen_hlth)
ment_hlth = st.slider("How many number of mentally unhealthy days you experience in a month?", min_value=0, max_value=30, value=5)
phys_hlth = st.slider("How many number of physically unhealthy days you experience in a month?", min_value=0, max_value=30, value=5)
diff_walk = st.selectbox("Enter your difficulty walking (0 = None, 1 = Severe)", [0, 1])
sex = st.selectbox("Select your gender", ["Female", "Male"])
if sex == "Female":
    sex = 0
else:
    sex = 1
persone_age = st.slider("Please specify your age", min_value=1, max_value=100, value=18)

def get_age_category(age):
    if 18 <= age <= 24:
        return 1
    elif 25 <= age <= 29:
        return 2
    elif 30 <= age <= 34:
        return 3
    elif 35 <= age <= 39:
        return 4
    elif 40 <= age <= 44:
        return 5
    elif 45 <= age <= 49:
        return 6
    elif 50 <= age <= 54:
        return 7
    elif 55 <= age <= 59:
        return 8
    elif 60 <= age <= 64:
        return 9
    elif 65 <= age <= 69:
        return 10
    elif 70 <= age <= 74:
        return 11
    elif 75 <= age <= 79:
        return 12
    elif age >= 80:
        return 13

age = get_age_category(persone_age)

# Model Selection
st.header("Select a Model")
model_choice = st.radio("Choose a model", ("DNN Model", "MLP Model", "Random Forest Model"))

# Convert inputs to numpy array and make a prediction
if st.button("Run Model"):
    # Prepare the input data for prediction
    input_data = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity,
                            fruits, veggies, hvy_alcohol, no_doc_cost, gen_hlth, ment_hlth,
                            phys_hlth, diff_walk, sex, age]])

    try:
        # Make a prediction using the selected model
        if model_choice == "DNN Model":
            prediction = mlp_model.predict(input_data)
        elif model_choice == "MLP Model":
            prediction = mlp_2_model.predict(input_data)
        elif model_choice == "Random Forest Model":
            prediction = rf_model.predict_proba(input_data)

        # Get the predicted class and label it as "No Diabetes", "Prediabetes", or "Diabetes"
        predicted_class = np.argmax(prediction, axis=1)[0]
        labels = ["No Diabetes", "Prediabetes", "Diabetes"]  # Adjust as per your classes
        prediction_label = labels[predicted_class]

        # Display the result
        st.write(f"The model predicts: {prediction_label}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")