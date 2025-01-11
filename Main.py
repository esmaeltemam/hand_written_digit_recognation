import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import pickle  # For saving the trained model

# Load MNIST data
def load_data():
    data = loadmat('mnist-original.mat')
    X = data['data'].T
    y = data['label'][0]
    return X, y

# Train the model (only run once)
def train_model():
    X, y = load_data()
    
    # Create a simple MLP classifier (can adjust parameters)
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=400, random_state=1)
    model.fit(X, y)
    
    # Save the model
    with open('digit_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved.")

# Predict a drawn digit (from the GUI)
def predict_digit(image_vector):
    # Load the trained model
    with open('digit_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make the prediction
    prediction = model.predict(image_vector)
    return prediction

# Example function to visualize an image (for testing)
def visualize_image(image_vector):
    plt.imshow(image_vector.reshape(28, 28), cmap='gray')
    plt.show()

if __name__ == "__main__":
    # Training the model (you only need to run this part once)
    # train_model()  # Uncomment this line to train and save the model

    # Use this for prediction and testing the model with the GUI
    #this the end
    pass
