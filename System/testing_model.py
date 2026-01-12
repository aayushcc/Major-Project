import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load the trained YOLOv11s model
MODEL_PATH = "best.pt"  # change to your YOLOv11s trained model path
model = YOLO(MODEL_PATH)

st.title("🚦 Vehicle Detection at Traffic Signals")
st.write("Upload a traffic signal image and detect vehicles using your YOLOv11s model.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    # Run YOLO detection
    results = model.predict(source=temp_path, save=True, conf=0.397)

    # Find the saved image (YOLO saves results in 'runs/detect')
    result_dir = results[0].save_dir
    detected_image_path = os.path.join(result_dir, os.listdir(result_dir)[0])

    # Display detection results
    st.subheader("Detection Results")
    st.image(detected_image_path, caption="Detected Vehicles", use_container_width=True)

    # Show counts
    vehicle_count = sum(1 for box in results[0].boxes if int(box.cls) in [0, 2, 3, 5, 7])  
    # Adjust class IDs according to your dataset labels
    st.write(f"**Detected Vehicles:** {vehicle_count}")
