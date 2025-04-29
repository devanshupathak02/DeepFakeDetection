"""
Enhanced Web Interface for Deepfake Detection
This standalone script provides an improved web interface using Gradio for the deepfake detection system.
"""

import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from tensorflow.keras import models
import time

# Import the DeepfakeDetector class (in a real scenario you'd import from the main file)
# For this example we'll redefine a simplified version
class SimpleDeepfakeDetector:
    def __init__(self, model_path):
        self.model = models.load_model(model_path)
        self.frame_size = (224, 224)
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from a video file"""
        temp_dir = tempfile.mkdtemp()
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        if frame_count <= max_frames:
            interval = 1
        else:
            interval = frame_count // max_frames
            
        count = 0
        frames_saved = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % interval == 0 and frames_saved < max_frames:
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
                frames_saved += 1
                
            count += 1
            
        cap.release()
        return frames
    
    def predict_image(self, img):
        """Predict whether an image is real or fake"""
        img = cv2.resize(img, self.frame_size)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        prediction = self.model.predict(img)[0][0]
        return prediction
    
    def predict_video(self, video_path):
        """Predict whether a video contains deepfakes"""
        frames = self.extract_frames(video_path)
        
        if not frames:
            return {"prediction": "Error", "confidence": 0.0, "message": "No frames extracted"}
            
        # Analyze each frame
        predictions = []
        for frame in frames:
            try:
                pred = self.predict_image(frame)
                predictions.append(pred)
            except Exception as e:
                print(f"Error processing frame: {e}")
                
        # Calculate the final prediction
        if not predictions:
            return {"prediction": "Error", "confidence": 0.0, "message": "Could not analyze frames"}
            
        avg_prediction = sum(predictions) / len(predictions)
        
        if avg_prediction > 0.5:
            return {"prediction": "Fake", "confidence": float(avg_prediction)}
        else:
            return {"prediction": "Real", "confidence": float(1 - avg_prediction)}
        
    def predict_media(self, media_path):
        """Predict whether a media file (image or video) is real or fake"""
        # Check if it's an image or video based on the extension
        ext = os.path.splitext(media_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = cv2.imread(media_path)
            pred = self.predict_image(img)
            if pred > 0.5:
                return {"prediction": "Fake", "confidence": float(pred)}
            else:
                return {"prediction": "Real", "confidence": float(1 - pred)}
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return self.predict_video(media_path)
        else:
            return {"prediction": "Error", "confidence": 0.0, "message": f"Unsupported file format: {ext}"}


def create_enhanced_web_interface(model_path):
    """Create an enhanced Gradio web interface for the deepfake detector"""
    
    # Initialize the detector
    detector = SimpleDeepfakeDetector(model_path)
    
    def process_media(file):
        start_time = time.time()
        
        # Create a temporary file to store the uploaded content
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        # Copy uploaded file to temp file
        with open(file.name, 'rb') as f_in:
            with open(temp_file.name, 'wb') as f_out:
                f_out.write(f_in.read())
        
        try:
            # Process the media file
            result = detector.predict_media(temp_file.name)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create the result message
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0) * 100
            message = result.get('message', '')
            
            # Create the output components
            result_label = f"{prediction}"
            confidence_value = confidence
            detail_text = f"Prediction: {prediction}\nConfidence: {confidence:.2f}%\nProcessing Time: {processing_time:.2f} seconds"
            
            if message:
                detail_text += f"\nMessage: {message}"
                
            # Determine the color based on prediction
            if prediction == "Fake":
                result_color = "red"
            elif prediction == "Real":
                result_color = "green"
            else:
                result_color = "gray"
                
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            return result_label, confidence_value, detail_text, result_color
            
        except Exception as e:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            return "Error", 0, f"Error processing file: {str(e)}", "gray"
    
    # Create the Gradio interface
    with gr.Blocks(title="Deepfake Detection System") as interface:
        gr.Markdown("# Deepfake Detection System")
        gr.Markdown("Upload an image or video to check if it's real or a deepfake.")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="Upload Image or Video")
                submit_btn = gr.Button("Analyze", variant="primary")
                
            with gr.Column(scale=3):
                with gr.Box():
                    with gr.Row():
                        result_label = gr.Label(label="Result", value="", elem_id="result_label")
                        confidence_meter = gr.Number(label="Confidence (%)", value=0, elem_id="confidence")
                    
                    detail_text = gr.TextArea(label="Details", value="", elem_id="details", interactive=False)
                
        # Example tabs
        with gr.Tabs():
            with gr.TabItem("How It Works"):
                gr.Markdown("""
                ## How Deepfake Detection Works
                
                This system analyzes images and videos to determine if they are authentic or artificially generated using AI.
                
                For videos, multiple frames are analyzed and the results are aggregated to produce a final prediction.
                
                The prediction includes:
                - Classification: Real or Fake
                - Confidence score: How certain the system is about the prediction
                - Processing time: How long it took to analyze the media
                
                ### Technologies Used
                - Deep learning with EfficientNet
                - Computer vision techniques
                - Frame analysis for videos
                """)
                
            with gr.TabItem("Tips"):
                gr.Markdown("""
                ## Tips for Best Results
                
                - **Image Quality**: Higher resolution images provide better detection results
                - **Video Length**: For videos, at least 5 seconds is recommended for accurate detection
                - **Face Visibility**: Clear view of faces improves detection accuracy
                - **Supported Formats**:
                  - Images: JPG, JPEG, PNG, BMP
                  - Videos: MP4, AVI, MOV, MKV
                """)
        
        # Set up the event handler
        submit_btn.click(
            fn=process_media,
            inputs=file_input,
            outputs=[result_label, confidence_meter, detail_text, result_label]
        )
        
    return interface

# Run the interface if this script is executed directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Web Interface')
    parser.add_argument('--model', type=str, default='deepfake_model.h5',
                      help='Path to the trained model file')
    
    args = parser.parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train a model first or provide the path to an existing model.")
        exit(1)
    
    # Create and launch the interface
    interface = create_enhanced_web_interface(args.model)
    interface.launch()
