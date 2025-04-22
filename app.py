import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # Using joblib to save/load the scaler

# --- Model Definition (Copied from your notebook) ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- Load Model and Scaler ---
# Ensure the model file 'dqn_model.pth' and the scaler file 'scaler_X.joblib'
# are in the same directory as app.py or provide the correct path.

# Define model parameters based on your training script
STATE_DIM = 5
ACTION_DIM = 2
MODEL_PATH = "dqn_model.pth" # Make sure this file exists
SCALER_PATH = "scaler_X.joblib" # You need to save the scaler during training

# Load the trained model state dictionary
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(STATE_DIM, ACTION_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # map_location ensures compatibility if trained on GPU and deployed on CPU
    model.to(device)
    model.eval() # Set the model to evaluation mode
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# --- Function to Save Scaler (Run this once after training in your notebook) ---
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib
#
# # Load your dataset as in the notebook
# df = pd.read_csv("predictive_maintenance.csv")
# features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
# X = df[features]
#
# # Fit the scaler
# scaler_X = MinMaxScaler()
# scaler_X.fit(X) # Fit on the original, unscaled training data features
#
# # Save the scaler
# joblib.dump(scaler_X, SCALER_PATH)
# print(f"Scaler saved to {SCALER_PATH}")


# Load the fitted scaler
try:
    scaler_X = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error(f"Error: Scaler file not found at {SCALER_PATH}. Please ensure you have saved the scaler using joblib after fitting it on your training data.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the scaler: {e}")
    st.stop()


# --- Prediction Function --- adapted for single input
def predict(model, scaler, input_data, device):
    # Convert input to numpy array and reshape for scaler
    input_array = np.array(input_data).reshape(1, -1)

    # Scale the input data
    input_scaled = scaler.transform(input_array)

    # Convert scaled data to tensor
    state = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension

    # Make prediction
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        q_values = model(state)
        action = torch.argmax(q_values, dim=1).item() # Get the action (0 or 1)

    return action

# --- Streamlit App Interface ---
st.title('Predictive Maintenance Failure Prediction')
st.write('Enter the sensor values to predict machine failure.')

# Input fields for features
st.sidebar.header('Input Sensor Data')
air_temp = st.sidebar.number_input('Air temperature [K]', value=298.1, format="%.1f")
process_temp = st.sidebar.number_input('Process temperature [K]', value=308.6, format="%.1f")
rot_speed = st.sidebar.number_input('Rotational speed [rpm]', value=1500)
torque = st.sidebar.number_input('Torque [Nm]', value=42.8, format="%.1f")
tool_wear = st.sidebar.number_input('Tool wear [min]', value=0)

# Prepare input data for prediction
input_features = [air_temp, process_temp, rot_speed, torque, tool_wear]

# Display input data
st.subheader("Input Data:")
input_df = pd.DataFrame([input_features], columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
st.dataframe(input_df)


# Predict button
if st.button('Predict Failure'):
    prediction = predict(model, scaler_X, input_features, device)
    st.subheader('Prediction:')
    if prediction == 0:
        st.success('Predicted: No Failure (0)')
    else:
        st.error('Predicted: Failure (1)')
