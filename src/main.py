import os
from src.preprocessing import preprocess_images
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image

def main():
    # Data preprocessing
    preprocess_images()

    # Build and train the model
    model = build_model()
    train_model(model)

    # Evaluate the model
    evaluate_model(model)

    # Make predictions
    input_image_path = input('Path of the image to be predicted: ')
    predict_image(model, input_image_path)

if __name__ == "__main__":
    main()
