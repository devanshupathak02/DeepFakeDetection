"""
Deepfake Detection Training Pipeline

This script provides a comprehensive training pipeline for the deepfake detection model,
including dataset preparation, training, evaluation, and model saving.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import pandas as pd
import cv2
from tqdm import tqdm
import argparse
import random
from pathlib import Path

# Setup GPU memory growth
gp
