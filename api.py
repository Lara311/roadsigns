from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = FastAPI()

# Load the ONNX model
onnx_model_path = 'yolov5s.onnx'
session = ort.InferenceSession(onnx_model_path)

def preprocess_image(image: Image.Image):
    image = image.resize((640, 640))  # Resize image to model input size
    image_np = np.array(image).astype(np.float32)  # Convert to numpy array
    image_np = np.transpose(image_np, [2, 0, 1])  # Change the order of dimensions
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Get input and output names from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run the model
    outputs = session.run([output_name], {input_name: input_tensor})
    
    # Process the model output (For simplicity, we return the raw output)
    return {"predictions": outputs[0].tolist()}
