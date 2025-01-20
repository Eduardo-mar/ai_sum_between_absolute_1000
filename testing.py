import tensorflow as tf
import numpy as np

charged_model = tf.keras.models.load_model('models/sum_2.h5')
# Check its architecture
charged_model.summary()

# Use the model to make predictions
X_new = [[0, -5], [100, 7], [32, 40], [25, -54], [1996, 1993], [3000, 6000]]  # New data for prediction
predictions = charged_model.predict(x=np.array(X_new), batch_size=None, verbose='auto', steps=None, callbacks=None)  # X_new is your input data for prediction
print(predictions)  # Output of the model
