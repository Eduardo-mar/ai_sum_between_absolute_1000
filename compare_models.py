import tensorflow as tf
import numpy as np
from visualizations import visualize_comparitive

def compare_models(model_paths, X_test, y_test):
    """
    Compare the performance of multiple trained models.
    
    Parameters:
    - model_paths: List of file paths to the saved models (in .h5 format).
    - X_test: The test dataset features.
    - y_test: The test dataset labels.
    """
    model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    results = []  # Store evaluation results
    
    for path in model_paths:
        # Load the model
        model = tf.keras.models.load_model(path)
        # Evaluate on the test data
        loss, mae = model.evaluate(X_test, y_test, verbose=0)  # Add metrics as needed
        results.append((loss, mae))
    
    # Convert results to NumPy array for easy plotting
    results = np.array(results)
    losses = results[:, 0]
    maes = results[:, 1]
    
    # Plot comparison
    visualize_comparitive(model_names, losses, maes)

# Example usage
# Paths to the saved models
model_paths = ["models/sum_1.h5", "models/sum_2.h5"]

# Example test dataset (replace with actual test data)
X_test = np.array([[0, -5], [100, 7], [32, 40], [25, -54], [1996, 1993]])  # Replace with your features
y_test = np.array([[-5], [107], [72], [-29], [3989]])  # Replace with your labels

# Compare the models
compare_models(model_paths, X_test, y_test)
