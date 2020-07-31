# practice17-transfer_learning
Using transfer learning method to build a model which distinguishes handwriting numbers with NN and CNN

In previous practices, we use ".add" to stack layers for models.
This time, we'd like to use another method called "Transfer Learning" to build up models.
Transfer Learning has a benefit that we can modify layer details or use some excellent layers built by other experts.
We use examples of handwriting numbers from MNIST datasets, and train the models by NN & CNN methods.
Unlike previous experience which models need to distinguish all 10 numbers(0-9), we only use front layers and modify output layer or fully-connected layer to generate expected outcomes.

Let's explore inside the code.
