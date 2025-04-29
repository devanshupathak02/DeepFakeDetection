# DeepFakeDetection
# Deepfake Detection System

This project provides a complete solution for detecting deepfakes in images and videos using deep learning techniques.

## Features

- Processes both images and video files
- Extracts frames from videos for analysis
- Uses an EfficientNetB0-based CNN for deepfake classification
- Provides confidence scores for predictions
- Includes a web interface for easy testing
- Works in near real-time for single images

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Gradio (for web interface)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

2. Install dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm gradio
```

## Dataset Preparation

The system expects a dataset organized in the following structure:
```
deepfake_dataset/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

You can use the following recommended datasets:
1. **FaceForensics++**: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
2. **Deepfake Detection Challenge Dataset (DFDC)**: [https://ai.facebook.com/datasets/dfdc/](https://ai.facebook.com/datasets/dfdc/)
3. **Celeb-DF**: [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)

## Usage

### Training Mode

Train the model using your prepared dataset:

```bash
python deepfake_detection.py --mode train --dataset path/to/dataset --epochs 15 --model my_model.h5
```

### Prediction Mode

Analyze a single image or video file:

```bash
python deepfake_detection.py --mode predict --model my_model.h5 --file path/to/media/file.jpg
```

### Web Interface Mode

Launch a user-friendly web interface:

```bash
python deepfake_detection.py --mode web --model my_model.h5
```

Then open your browser and go to the URL displayed in the terminal (usually http://127.0.0.1:7860).

## Model Architecture

The system uses an EfficientNetB0 pretrained on ImageNet as its base model, with additional layers:

- Global Average Pooling
- Batch Normalization
- Dense layers with dropout for regularization
- Binary classification output (Real/Fake)

The training process includes:
1. Initial training with frozen base model
2. Fine-tuning by unfreezing the last 20 layers of the base model

## Performance Optimization

For better performance:
- GPU acceleration is automatically used when available
- Frame extraction from videos is limited to 30 evenly distributed frames
- Data augmentation is applied during training to improve generalization

## Contributing

Contributions to improve the system are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
