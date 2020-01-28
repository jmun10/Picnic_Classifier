import tensorflow as tf
import matplotlib
from tensorflow import keras
import numpy as np
import importlib
import tfrFile
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_dir=r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy"

from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(train_dir)
model_trainer.trainModel(num_objects=26, num_experiments=150, enhance_data=True, batch_size=64, show_network_summary=True)

