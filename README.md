# FlowerClassifier-CNN
This project represents my first approach to convolutional neural networks (CNNs) and neural networks in general. I developed FlowerClassifier-CNN as a beginner's exploration into deep learning, focusing on classifying flower species through image recognition.

## Features
- **Flower Recognition**: Classifies uploaded flower images into species categories.
- **Flask Web Application**: A simple Flask server enables an interactive, user-friendly web interface for image uploads.

### Flower Classes
The FlowerClassifier-CNN is trained to recognize the following flower species:
- **Lilly**
- **Lotus**
- **Orchid**
- **Sunflower**
- **Tulip**
  
## Project Overview
This project is my first attempt at building a convolutional neural network (CNN) for image classification. I used the [5 Flower Types Classification Dataset](https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset)
, which contains 1,000 images for each of the five flower classes: Lilly, Lotus, Orchid, Sunflower, and Tulip.

### Model Architecture
The model is built using TensorFlow and Keras with the following architecture:

1. **Input Layer**: The model accepts input images with a size of 128x128 pixels and 3 color channels (RGB). <br><br>
2. **Convolutional Layers**: The architecture includes multiple convolutional layers with increasing filter sizes. The initial layers use 32 and 64 filters, progressing to 128 and 256 filters in deeper layers. This design helps the model learn complex features from the images while maintaining spatial information.<br><br>
3. **Activation Functions**: Each convolutional layer employs the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity and aids in learning complex patterns.<br><br>
4. **Batch Normalization**: After each convolutional layer, batch normalization is applied to stabilize and speed up the training process by normalizing the output of the previous layer.<br><br>
5. **Pooling Layers**: Max pooling layers are utilized to reduce the spatial dimensions of the feature maps, which helps in reducing computation and controlling overfitting.<br><br>
6. **Flattening**: After the convolutional and pooling layers, the model flattens the feature maps into a single vector to prepare for the fully connected (dense) layers.<br><br>
7. **Dense Layers**: The model features two dense layers, each with 128 neurons and ReLU activation. Dropout regularization (set at 40%) is applied after these layers to prevent overfitting by randomly setting a portion of the neurons to zero during training.<br><br>
8. **Output Layer**: The final layer is a dense layer with five neurons, each representing a different flower class. This layer uses a softmax activation function to produce a probability distribution over the classes.<br><br>
9. **L2 Regularization**: L2 regularization is applied to the convolutional and dense layers to penalize large weights and further mitigate overfitting.<br><br>
10. **Data Augmentation**: To enhance the diversity of the training dataset and reduce overfitting, data augmentation techniques such as rotation, zoom, and flipping were employed during training.<br><br>
11. **Compilation**: The model is compiled using the Adam optimizer with a reduced learning rate for stability, and it employs categorical cross-entropy as the loss function, which is suitable for multi-class classification tasks.<br><br>
12. **Callbacks**: During training, callbacks such as learning rate reduction on plateaus and early stopping are implemented to optimize training efficiency and prevent overfitting.

## Optimization Insights
* I noticed that the optimization(Adam optimizer) step played a crucial role in the model's performance. In the beginning, the model often overshot optimal solutions due to an aggressive learning rate. To address this, I modified the learning rate dynamically during training, which helped stabilize the convergence and improve overall accuracy.

## Challenges
* The most challenging aspect of this project was addressing the overfitting problem. Despite implementing dropout, L2 regularization, and data augmentation, the model still tended to overfit, particularly during the initial training phases. I experimented with various architectures and hyperparameters to mitigate this issue, but achieving a perfect balance between training and validation accuracy remained a challenge.

### After training for 40 epochs, I achieved the following results:
- Training Accuracy: 95.57%
- Training Loss: 0.5270
- Validation Accuracy: 91.40%
- Validation Loss: 0.6916
  
Throughout the project, I experimented with various architectures and hyperparameters, which made it a great learning experience. While I’m pleased with the results, I acknowledge that the model is not perfect, and I’m open to suggestions for improvements.

## How to Run the Application

To get started with the FlowerClassifier-CNN web application, make sure you have **Flask** and **TensorFlow** installed. Then, follow these steps to launch the server:

### Running the Server
- Start the Flask server by running the following command in your terminal:
  ```bash
  python app.py
- Open a browser and go to http://127.0.0.1:5000 to access the application. Here, you can upload an image to classify the flower species.

## Quick Demo
A quick demo follows below to showcase the application in action.

https://github.com/user-attachments/assets/6c54ba6b-24e1-47e6-9783-5c1bb420526c

