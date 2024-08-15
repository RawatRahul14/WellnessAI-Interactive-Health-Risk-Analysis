from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Flatten
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

import streamlit as st
import numpy as np
import tensorflow as tf

# Define the model architecture
input_shape = (21,)  # Adjust input shape as per your model
mlp_model = MLP_model(input_shape)

input_shape_cnn = (21, 1, 1)
cnn_model = CNN_model(input_shape_cnn)

# Load pre-trained model weights
try:
    mlp_model.load_weights("Dataset\Model_selection\model_mlp.h5")
    cnn_model.load_weights("Dataset\Model_selection\model_cnn.h5")
    st.success("Model weights loaded successfully.")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Set the title of the app
st.title("Diabetes Prediction UI")

# User Inputs
st.header("Enter Your Health Information")

# Input fields corresponding to the dataset columns
high_bp = st.number_input("High Blood Pressure (HighBP)", min_value=0.0, max_value=1.0, value=0.5)
high_chol = st.number_input("High Cholesterol (HighChol)", min_value=0.0, max_value=1.0, value=0.5)
chol_check = st.selectbox("Cholesterol Check (CholCheck)", [0, 1])
height = st.number_input("Height (in cms)", min_value=10.0, max_value=200.0, value=100.0)
weight = st.number_input("Weight (in kgs)", min_value=0.0, max_value=120.0, value=30.0)
bmi = (weight/height ** 2) * 10000
smoker = st.selectbox("Smoker", [0, 1])
stroke = st.selectbox("Stroke", [0, 1])
heart_disease = st.selectbox("Heart Disease or Attack (HeartDiseaseorAttack)", [0, 1])
phys_activity = st.selectbox("Physical Activity (PhysActivity)", [0, 1])
fruits = st.selectbox("Fruits", [0, 1])
veggies = st.selectbox("Vegetables (Veggies)", [0, 1])
hvy_alcohol = st.selectbox("Heavy Alcohol Consumption (HvyAlcoholConsump)", [0, 1])
any_healthcare = st.selectbox("Any Healthcare", [0, 1])
no_doc_cost = st.selectbox("No Doctor Because of Cost (NoDocbcCost)", [0, 1])
gen_hlth = st.slider("General Health (GenHlth)", min_value=1, max_value=5, value=3)
ment_hlth = st.slider("Mental Health (MentHlth)", min_value=0, max_value=30, value=5)
phys_hlth = st.slider("Physical Health (PhysHlth)", min_value=0, max_value=30, value=5)
diff_walk = st.selectbox("Difficulty Walking (DiffWalk)", [0, 1])
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
age = st.slider("Age Category (Age)", min_value=1, max_value=13, value=7)  # Assuming age categories are 1-13
education = st.slider("Education Level (Education)", min_value=1, max_value=6, value=4)
income = st.slider("Income Level (Income)", min_value=1, max_value=8, value=4)

# Model Selection
st.header("Select a Deep Learning Model")
model_choice = st.radio("Choose a model", ("MLP", "CNN"))

# Display the selected model and inputs
st.subheader("Selected Model:")
st.write(f"Model: {model_choice}")

st.subheader("Your Inputs:")
st.write({
    "HighBP": high_bp, "HighChol": high_chol, "CholCheck": chol_check, "BMI": bmi, "Smoker": smoker,
    "Stroke": stroke, "HeartDiseaseorAttack": heart_disease, "PhysActivity": phys_activity,
    "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": hvy_alcohol,
    "AnyHealthcare": any_healthcare, "NoDocbcCost": no_doc_cost, "GenHlth": gen_hlth,
    "MentHlth": ment_hlth, "PhysHlth": phys_hlth, "DiffWalk": diff_walk, "Sex": sex,
    "Age": age, "Education": education, "Income": income
})

# Convert inputs to numpy array and make a prediction
if st.button("Run Model"):
    # Prepare the input data for prediction
    input_data = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity,
                            fruits, veggies, hvy_alcohol, any_healthcare, no_doc_cost, gen_hlth, ment_hlth,
                            phys_hlth, diff_walk, sex, age, education, income]])
    
    try:
        # Make a prediction using the model
        if model_choice == "MLP":
            prediction = mlp_model.predict(input_data)
        elif model_choice == "CNN":
            input_data = input_data.reshape(-1, input_data.shape[1], 1, 1)
            prediction = cnn_model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        labels = ["No Diabetes", "Prediabetes", "Diabetes"]  # Adjust as per your classes
        prediction_label = labels[predicted_class]
        
        # Display the result
        st.write(f"The model predicts: {prediction_label}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
