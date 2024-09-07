# COMPUTER-VISION-AND-DEEP-LEARNING

# Problem Statement: 
this project is to classify galaxies into distinct types such as elliptical, irregular, and spiral using a deep learning approach. Given the vast number of astronomical images available due to advancements in telescopic technology, manual classification is impractical. Automated classification using machine learning and deep learning methods addresses this challenge, enabling more accurate and faster analysis of galaxy images.

# Abstract:
This project implements a deep convolutional neural network (CNN) to classify galaxies into different morphological categories.
The model architecture consists of multiple layers, including convolutional, fully connected, and dropout layers, trained on a dataset of over 1360 images.
The system achieved an impressive testing accuracy of 97.32%, outperforming other classification methods found in related research. This project explores the effectiveness of deep learning in astronomical image classification.

# Introduction:
Understanding the classification of galaxies is crucial for unraveling the mysteries of the universe's origin and evolution. Traditionally, astronomers have relied on manual inspection and classification of galaxy images, which is both time-consuming and limited in scale. With the advent of deep learning and automated systems, it's now possible to classify galaxies in large datasets more accurately and efficiently. This project focuses on creating a CNN model that automates the classification process, using features extracted directly from the raw images of galaxies.

# How we built it?
* Dataset: We utilized a dataset from Kaggle containing images of stars and galaxies.
  
* Preprocessing: Images were resized, normalized, and labeled to prepare them for training.
  
* Model: A Convolutional Neural Network (CNN) was designed with 8 layers, including:
  
* Convolutional layersfor feature extraction
  
* MaxPooling layers to reduce dimensionality
  
* Fully connected layers for classification
  
* Training: The model was trained using TensorFlow and Keras, with dropout layers to prevent overfitting. We used ReLU and sigmoid activations in different layers.
  
* Evaluation: The model was evaluated on a test dataset, and results were visualized using confusion matrices and heatmaps to interpret the classification results.

# Advantages of this system:
1).High Accuracy: The model achieved 97.32% testing accuracy, which is higher than previous research efforts.

2).Efficiency: Automating galaxy classification significantly reduces the time taken to analyze large datasets, which would be impractical for human classifiers.

3).Scalability: The system can be applied to a vast number of images from telescopic surveys, handling large-scale astronomical data with ease.

# Shortcomings of this system:
1).Generalization: While the model performs well on the given dataset, it may not generalize as effectively to new datasets or images with noise and distortions.

2).Limited Dataset: The dataset used for training is relatively small (~1360 images), and larger, more diverse datasets may be needed for better generalization.

3).Black-box Nature: The CNN architecture, like other deep learning models, operates as a black-box, making it difficult to interpret how specific features influence classification decisions.

# Future Scope:
* Dataset Expansion: Incorporating larger and more diverse datasets, including images from newer telescopes, could improve model generalization.
* Improved Models: Exploring advanced architectures such as ResNet, DenseNet, or Transformer models could further boost classification accuracy.
* Multimodal Analysis: Integrating spectral data with images could provide a more holistic approach to galaxy classification.
* Real-Time Classification: The system could be adapted to classify galaxies in real-time as new astronomical data is collected.

# What we have learnt:
We have learnt about many new libraries and how to use them like numpy, tensorflow, keras. The intresting part was to learn about how to create a model based on Convolutional Neural Network, and dive deeper into the world of Artifical Intelligence. Got to know about that there exist such dataset named keras.
