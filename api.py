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
    try:
        # Resize the image to 640x640
        image = image.resize((640, 640))  
        image_np = np.array(image).astype(np.float32)  # Convert to numpy array
        image_np = np.transpose(image_np, [2, 0, 1])  # Convert to CHW format
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension of 1

        # Repeat the image to create a batch of 16 (or adjust the batch size as needed)
        image_np = np.repeat(image_np, 16, axis=0)  # Repeat along the batch dimension
        
        return image_np
    except Exception as e:
        print(f"Error in preprocessing image: {str(e)}")
        raise

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

@app.get("/packages")
async def list_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    return {"installed_packages": result.stdout.decode('utf-8').split('\n')}
