# Iris-flower-classification-with-Neural Networks

This project aims to classify observations from the famous Iris dataset into one of three species: Setosa, Versicolor, or Virginica. We use a neural network built with PyTorch to make predictions based on sepal length, sepal width, petal length, and petal width.

**Project Overview**

The goal is to classify each observation into one of three species based on the measurement of sepal and petal dimensions. The project follows a typical deep learning pipeline, including dataset preparation, model building, forward propagation, loss computation, backpropagation, and prediction.

**Dataset**

We use the well-known Iris dataset, which contains 150 observations of iris flowers, each with four features: Sepal Length, Sepal Width, Petal Length, Petal Width

The dataset is publicly available at this link http://bit.ly/IrisDataset.

**Steps followed**
- Step 1: Dataset preparation

- Step 2: Building the model

- Step 3: Forward propagation

- Step 4: Loss computation

- Step 5: Backpropagation

- Step 6: Make predictions

  **Neural Network Architecture**
  
The neural network is composed of:

- Input Layer: 4 input nodes (one for each feature)
- Hidden Layer: 8 nodes with ReLU activation
- Output Layer: 3 output nodes (one for each species), using softmax activation for multi-class classification

 ** Training Process**
 
- Forward Propagation: Pass the input data through the model to get predictions.
- Loss Computation: We use Cross-Entropy Loss to compute the difference between predicted and actual labels.
- Backpropagation: Compute gradients and update model parameters using Stochastic Gradient Descent (SGD).
- Epochs: We train the model for 100 epochs, monitoring the loss at regular intervals.

**Results**

- Training Loss: Reduced from 0.92 to 0.29 over 100 epochs.
- Accuracy: Achieved 86.67% accuracy on the test set.

**Future Improvements**

- Implement hyperparameter tuning to optimize the model further.
- Experiment with different architectures and activation functions.
- Use techniques like dropout to prevent overfitting.
