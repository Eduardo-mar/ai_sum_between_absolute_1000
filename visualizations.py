import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualize weights of the layers
def visualize_weights(model):
    """
    Visualize the weights of all layers (hidden and output) in the neural network.
    """
    num_layers = len(model.layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))  # Create subplots for each layer
    
    if num_layers == 1:  # If there's only one layer
        axes = [axes]  # Ensure axes is iterable

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) > 0:  # Skip layers without weights (e.g., input layer)
            W = weights[0]
            sns.heatmap(W, annot=True, cmap='coolwarm', cbar=True, ax=axes[i])
            axes[i].set_title(f'Weights: {layer.name}')
            axes[i].set_xlabel('Input Features')
            axes[i].set_ylabel('Neurons')
        else:
            axes[i].set_visible(False)  # Hide subplot if no weights

    plt.tight_layout()
    plt.show()

# Visualize the biases of the layers
def visualize_bias(model, threshold=10):
    """
    Visualizes the biases for every layer in the neural network (excluding the input layer).
    Each layer's biases are displayed as a separate bar plot with optional value annotations.
    
    Parameters:
    - model: The trained neural network model.
    - threshold: Bias values greater than this threshold will be annotated on the bars.
    """
    layer_biases = []
    
    # Extract biases from each layer
    for i, layer in enumerate(model.layers):
        weights_and_biases = layer.get_weights()  # Get weights and biases
        if len(weights_and_biases) == 2:  # Check if the layer has biases
            biases = weights_and_biases[1]  # Biases are the second element
            layer_biases.append(biases)
        else:
            layer_biases.append(None)  # No biases for this layer
    
    # Plot biases
    num_layers = len(layer_biases)
    plt.figure(figsize=(12, 8))
    
    for i, biases in enumerate(layer_biases):
        if biases is not None:  # Plot only layers with biases
            plt.subplot(1, num_layers, i + 1)
            bars = plt.bar(range(len(biases)), biases, color='skyblue')
            plt.title(f"Layer {i+1} Biases")
            plt.xlabel("Neuron")
            plt.ylabel("Bias Value")
            plt.xticks(range(len(biases)))
            
            # Add bias values above bars if greater than the threshold
            for j, bar in enumerate(bars):
                if biases[j] > threshold:  # Annotate only if bias > threshold
                    plt.text(bar.get_x() + bar.get_width() / 2, 
                             bar.get_height() + 0.5,  # Position above bar
                             f"{biases[j]:.2f}",  # Bias value with 2 decimals
                             ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

# Visualize the structure of the neural network
def visualize_neural_net(model):
    """
    Visualizes the structure of a neural network with nodes and connections.
    Each layer is represented with circles, with connections drawn between layers.
    """
    # Determine layer sizes
    layer_sizes = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):  # For Dense layers, 'units' gives the number of neurons
            layer_sizes.append(layer.units)
    
    # Add the input layer size explicitly
    input_size = model.input_shape[-1]  # Get the input shape size (e.g., 1 for input_shape=(1,))
    layer_sizes.insert(0, input_size)   # Add input size as the first layer
    
    print(f"Layer sizes: {layer_sizes}")  # Debugging: Print layer sizes
    
    # Create a figure for the plot
    plt.figure(figsize=(12, 8))
    
    # Maximum number of neurons in a layer
    max_neurons = max(layer_sizes)
    
    # Define neuron positions for each layer
    neuron_positions = []
    for i, layer_size in enumerate(layer_sizes):
        x = np.full(layer_size, i)  # X-coordinates of the layer
        y = np.linspace(0, max_neurons - 1, layer_size)  # Evenly spaced Y-coordinates
        neuron_positions.append((x, y))
        
        # Plot the neurons
        plt.scatter(x, y, s=800, zorder=5, label=f"Layer {i+1}")
        for j in range(layer_size):
            plt.text(x[j], y[j], f"{j+1}", ha="center", va="center", color="black", zorder=10)
    
    # Draw connections between layers
    for i in range(len(neuron_positions) - 1):
        x1, y1 = neuron_positions[i]
        x2, y2 = neuron_positions[i + 1]
        for start_y in y1:
            for end_y in y2:
                plt.plot([x1[0], x2[0]], [start_y, end_y], color='gray', alpha=0.5)
    
    # Add labels and title
    plt.xticks(range(len(layer_sizes)), [f"Layer {i+1}" for i in range(len(layer_sizes))], fontsize=12)
    plt.yticks([])  # Hide Y-axis ticks
    plt.title("Neural Network Architecture", fontsize=16)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()  # Ensure everything fits
    plt.show()

# Visualize the loss curve
def visualize_loss_curve(history):
    """
    Visualizes the loss curve of a neural network during training.
    
    Parameters:
    - history: The training history of the neural network model.
    """
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, color='steelblue', marker='o', linestyle='-', linewidth=2)
    plt.title('Loss Function Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(8, 6))

# plt.plot(history.history['loss'], label='Loss', marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Function Over Epochs')
# plt.legend()
# plt.show()