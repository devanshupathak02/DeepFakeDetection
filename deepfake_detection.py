"""
Deepfake Detection System
========================
This code provides a complete implementation for detecting deepfakes in images and videos.
The system includes:
1. Frame extraction from videos
2. A deep learning model for deepfake detection
3. A web interface for uploading and analyzing media
4. Command-line tool functionality
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class DeepfakeDetector:
    def __init__(self, model_path=None):
        """Initialize the deepfake detector with an optional pre-trained model"""
        self.frame_size = (224, 224)  # Standard input size for many CNN architectures
        self.batch_size = 32
        
        if model_path and os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.model = models.load_model(model_path)
        else:
            print("Creating new model")
            self.model = self._build_model()
            
    def _build_model(self):
        """Build a deepfake detection model based on EfficientNetB0 with additional layers"""
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification: real (0) or fake (1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def extract_frames(self, video_path, output_dir, max_frames=30):
        """Extract frames from a video file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval to extract max_frames evenly distributed
        if frame_count <= max_frames:
            interval = 1
        else:
            interval = frame_count // max_frames
            
        frames_saved = 0
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % interval == 0 and frames_saved < max_frames:
                frame_path = os.path.join(output_dir, f"frame_{frames_saved:03d}.jpg")
                frame = cv2.resize(frame, self.frame_size)
                cv2.imwrite(frame_path, frame)
                frames_saved += 1
                
            count += 1
            
        cap.release()
        return frames_saved
    
    def prepare_training_data(self, real_dir, fake_dir):
        """Prepare image data for training from directories containing real and fake images"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            validation_split=0.2  # 20% for validation
        )
        
        # Only rescaling for validation
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.dirname(real_dir),  # Parent directory containing 'real' and 'fake' subdirectories
            target_size=self.frame_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        # Load validation data
        validation_generator = valid_datagen.flow_from_directory(
            directory=os.path.dirname(real_dir),  # Parent directory containing 'real' and 'fake' subdirectories
            target_size=self.frame_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=10):
        """Train the deepfake detection model"""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=[early_stopping, reduce_lr]
        )
        
        return history
    
    def fine_tune(self, train_generator, validation_generator, epochs=5):
        """Fine-tune the model by unfreezing some of the base model layers"""
        # Unfreeze the last 20 layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all the layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Continue training
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size
        )
        
        return history
    
    def save_model(self, model_path):
        """Save the trained model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
    def predict_image(self, image_path):
        """Predict whether a single image is real or fake"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        img = cv2.resize(img, self.frame_size)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        prediction = self.model.predict(img)[0][0]
        return {"prediction": "Fake" if prediction > 0.5 else "Real", 
                "confidence": float(prediction) if prediction > 0.5 else float(1 - prediction)}
    
    def predict_video(self, video_path, temp_dir="temp_frames"):
        """Predict whether a video contains deepfakes by analyzing frames"""
        # Create temporary directory for frames
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Extract frames
        num_frames = self.extract_frames(video_path, temp_dir)
        
        if num_frames == 0:
            return {"prediction": "Error", "confidence": 0.0, "message": "No frames extracted"}
            
        # Analyze each frame
        predictions = []
        for i in range(num_frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                try:
                    result = self.predict_image(frame_path)
                    predictions.append(1 if result["prediction"] == "Fake" else 0)
                except Exception as e:
                    print(f"Error processing frame {i}: {e}")
                    
        # Calculate the final prediction
        if len(predictions) == 0:
            return {"prediction": "Error", "confidence": 0.0, "message": "Could not analyze frames"}
            
        fake_ratio = sum(predictions) / len(predictions)
        
        # Clean up temporary frames
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
            
        if fake_ratio > 0.5:
            return {"prediction": "Fake", "confidence": float(fake_ratio)}
        else:
            return {"prediction": "Real", "confidence": float(1 - fake_ratio)}
        
    def predict_media(self, media_path):
        """Predict whether a media file (image or video) is real or fake"""
        # Check if the file exists
        if not os.path.exists(media_path):
            return {"prediction": "Error", "confidence": 0.0, "message": f"File not found: {media_path}"}
            
        # Check if it's an image or video based on the extension
        ext = os.path.splitext(media_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self.predict_image(media_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return self.predict_video(media_path)
        else:
            return {"prediction": "Error", "confidence": 0.0, "message": f"Unsupported file format: {ext}"}


def prepare_example_dataset(base_dir='deepfake_dataset'):
    """
    Function to prepare a simulated dataset structure for demonstration
    In a real scenario, you would download and use actual datasets
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create directory structure
    real_dir = os.path.join(base_dir, 'real')
    fake_dir = os.path.join(base_dir, 'fake')
    
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
    if not os.path.exists(fake_dir):
        os.makedirs(fake_dir)
        
    print(f"Created dataset directory structure at {base_dir}")
    print(f"Put real images in: {real_dir}")
    print(f"Put fake images in: {fake_dir}")
    
    return real_dir, fake_dir


def plot_training_history(history):
    """Plot training history metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def create_web_interface(detector):
    """Create a Gradio web interface for the deepfake detector"""
    def predict_file(file):
        try:
            result = detector.predict_media(file.name)
            confidence = result.get('confidence', 0) * 100
            return f"{result['prediction']} (Confidence: {confidence:.2f}%)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=predict_file,
        inputs=gr.File(label="Upload Image or Video"),
        outputs=gr.Textbox(label="Prediction"),
        title="Deepfake Detection System",
        description="Upload an image or video to check if it's real or a deepfake.",
        examples=[
            # You would add example files here in a real application
        ],
        allow_flagging="never"
    )
    
    return interface


def main():
    """Main function to run the deepfake detection system"""
    parser = argparse.ArgumentParser(description='Deepfake Detection System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'web'],
                        help='Mode to run the system in (train, predict, web)')
    parser.add_argument('--model', type=str, default='deepfake_model.h5',
                        help='Path to save/load the model')
    parser.add_argument('--dataset', type=str, default='deepfake_dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--file', type=str, help='Media file for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Set up dataset directories
        if not os.path.exists(args.dataset):
            real_dir, fake_dir = prepare_example_dataset(args.dataset)
            print("Please populate the dataset directories before training.")
            return
        else:
            real_dir = os.path.join(args.dataset, 'real')
            fake_dir = os.path.join(args.dataset, 'fake')
            
            if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
                print(f"Dataset directory structure is incorrect. Expected '{real_dir}' and '{fake_dir}'")
                real_dir, fake_dir = prepare_example_dataset(args.dataset)
                print("Please populate the dataset directories before training.")
                return
        
        # Initialize detector and train
        detector = DeepfakeDetector()
        train_gen, valid_gen = detector.prepare_training_data(real_dir, fake_dir)
        
        print("Starting initial training...")
        history = detector.train(train_gen, valid_gen, epochs=args.epochs)
        
        print("Starting fine-tuning...")
        fine_tune_history = detector.fine_tune(train_gen, valid_gen, epochs=min(5, args.epochs))
        
        # Save model
        detector.save_model(args.model)
        
        # Plot training history
        plot_training_history(history)
        
    elif args.mode == 'predict':
        if not args.file:
            print("Please provide a file to analyze with --file")
            return
            
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
            
        detector = DeepfakeDetector(model_path=args.model)
        result = detector.predict_media(args.file)
        
        print("\n=== Deepfake Detection Result ===")
        print(f"File: {args.file}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result.get('confidence', 0) * 100:.2f}%")
        if 'message' in result:
            print(f"Message: {result['message']}")
            
    elif args.mode == 'web':
        detector = DeepfakeDetector(model_path=args.model)
        interface = create_web_interface(detector)
        interface.launch()
        

if __name__ == "__main__":
    main()
