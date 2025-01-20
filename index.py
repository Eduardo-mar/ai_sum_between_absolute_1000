import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from visualizations import *

# Step 1: Import the dataset
data = pd.read_csv('dataset.csv')
print(data.head())  # Displays the first few rows of the dataset
print(data.columns)  # Lists all column names in the dataset

# Assuming your dataset has columns 'Feature' and 'Target'
X = data[['a', 'b']].values # Extract and reshape the feature column
y = data['c'].values.reshape(-1, 1)  # Extract and reshape the target column
print("Features (X):", X)
print("Labels (y):", y)
# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, y_train)

# Step 3: Create a simple neural network
model = Sequential([
    Dense(5, input_shape=(2,), activation='relu'),  # Hidden layer with 10 neurons
    # Dense(10, input_dim=10, activation='relu'),  # Hidden layer with 10 neurons
    Dense(10, activation='relu'),  # Hidden layer with 10 neurons
    Dense(5, activation='relu'),  # Hidden layer with 10 neurons
    Dense(1, activation='linear')  # Output layer with 1 neuron
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 4: Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# Step 5: Evaluate the model
print("\nEvaluating the model on test data...")
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Step 6: Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Display some predictions
for i in range(5):
    print(f"Input: {X_test[i]}, True: {y_test[i]}, Predicted: {y_pred[i]}")

# Step 7: Visualize Architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
print("\nModel architecture saved as 'model_architecture.png'.")

# Step 8: Use Visualization Functions
visualize_weights(model)

visualize_bias(model, threshold=10)

visualize_neural_net(model)

# Step 9: Plot the Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Over Epochs')
plt.legend()
plt.show()

# Step 10: Summarize Activation Functions and Final Model State
print("Model Summary:")
model.summary()

print("\nActivation Functions by Layer:")
for layer in model.layers:
    print(f"Layer {layer.name}: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'Linear'}")

# Final Loss and Metric Values
final_loss = history.history['loss'][-1]
final_mae = history.history['mae'][-1]
print(f"\nFinal Loss: {final_loss:.4f}")
print(f"Final MAE: {final_mae:.4f}")

# Save the entire model as a SavedModel.
model.save('models/sum_2.h5')  # Save the model to a file
