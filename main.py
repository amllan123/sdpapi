from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import uvicorn
import joblib

# Load the models
model_wave = tf.keras.models.load_model("Wave_parkinson_disease_detection_model.h5")
model_vgg16 = tf.keras.models.load_model("parkinson_disease_detection_VGG16_final.h5")
model_load=load_model("bidirectional_lstm_model.h5")
model_blstm = load_model("bidirectional_lstm_model.h5")
scl= joblib.load("scaler.pkl")
# Initialize the FastAPI app
app = FastAPI()

# Define the response structure
class PredictionResponse(BaseModel):
    prediction: bool  # Assuming the prediction is boolean for both models

class InputData(BaseModel):
    x: float
    y: float
    z: float
    pressure_angle: float
    grip_angle: float
    timestamp:float


# Function to preprocess the input image for the wave model
def preprocess_image_wave(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize for the first model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    if len(image_array.shape) == 2 or image_array.shape[2] == 1:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to preprocess the input image for the VGG16 model
def preprocess_image_vgg16(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize for the second model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the prediction endpoint for the wave model
@app.post("/predict_wave", response_model=PredictionResponse)
async def predict_wave(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        input_data = preprocess_image_wave(image)
        prediction = model_wave.predict(input_data)
        Net_Prediction_Result = prediction[0][0] <= prediction[0][1]
        accuracy = 0
        if prediction[0][0]<prediction[0][1]:
            accuracy = prediction[0][1]*100
        else:
            accuracy = prediction[0][0]*100
        return {"prediction":Net_Prediction_Result, "accuracy":accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the prediction endpoint for the VGG16 model
@app.post("/predict_vgg16", response_model=PredictionResponse)
async def predict_vgg16(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        input_data = preprocess_image_vgg16(image)
        prediction = model_vgg16.predict(input_data)
        Net_Prediction_Result = prediction[0][0] <= prediction[0][1]
        return PredictionResponse(prediction=Net_Prediction_Result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict_bltsm")
def predict(data: InputData):
    try:
        # Extract input data
        X = np.array([[data.x, data.y, data.z, data.pressure_angle, data.grip_angle,data.timestamp]])
        
        # Scale the input data
        
        X_scaled = scl.transform(X)
        
        # Reshape the data for RNN input
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        # Make prediction
        prediction = model_load.predict(X_reshaped)
        result=False
        if prediction > 0.4:
            result=True
        # Return the prediction as a list
        return {"prediction": float(prediction[0][0])*100, "result":result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_combined", response_model=PredictionResponse)
async def predict_combined(
    wave_file: UploadFile = File(...), 
    spiral_file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    z: float = Form(...),
    pressure_angle: float = Form(...),
    grip_angle: float = Form(...),
    timestamp: float = Form(...)
):
    try:
        # Preprocess image for wave model
        wave_image_data = await wave_file.read()
        wave_image = Image.open(io.BytesIO(wave_image_data))
        input_data_wave = preprocess_image_wave(wave_image)
        prediction_wave = model_wave.predict(input_data_wave)
        result_wave = prediction_wave[0][0] <= prediction_wave[0][1]
        accuracy_wave = max(prediction_wave[0]) * 100

        # Preprocess image for VGG16 model
        spiral_image_data = await spiral_file.read()
        spiral_image = Image.open(io.BytesIO(spiral_image_data))
        input_data_vgg16 = preprocess_image_vgg16(spiral_image)
        prediction_vgg16 = model_vgg16.predict(input_data_vgg16)
        result_vgg16 = prediction_vgg16[0][0] <= prediction_vgg16[0][1]
        accuracy_vgg16 = max(prediction_vgg16[0]) * 100

        # Preprocess data for BLSTM model
        X = np.array([[x, y, z, pressure_angle, grip_angle, timestamp]])
        X_scaled = scl.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        prediction_blstm = model_blstm.predict(X_reshaped)
        result_blstm = prediction_blstm[0][0] > 0.4
        accuracy_blstm = float(prediction_blstm[0][0]) * 100

        # Normalize accuracies
        total_accuracy = accuracy_wave + accuracy_vgg16 + accuracy_blstm
        weights = np.array([accuracy_wave, accuracy_vgg16, accuracy_blstm]) / total_accuracy

        # Combine predictions into a single array
        all_preds = np.array([result_wave, result_vgg16, result_blstm])

        # Calculate the weighted average of the predictions
        weighted_preds = np.dot(weights, all_preds)
        final_pred = (weighted_preds > 0.5).astype(int)

        return {"prediction": bool(final_pred), "accuracy": float(weighted_preds * 100)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
