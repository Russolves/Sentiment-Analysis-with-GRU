# Sentiment-Analysis-with-GRU
This repository contains three variants of a Sentiment Analysis model that uses a GRU (Gated Recurrent Unit) to predict the sentiment of a given text as either positive or negative. The models were built using PyTorch, and the training and testing data came from DLStudio

The three variants included are
- customGRU
- torch.nn.GRU
- Bidirectional GRU

# Dataset
The training and testing data sets are contained in .tar.gz files, each containing positive and negative review data with the corresponding category label (positive or negative). The data is preprocessed, tokenized into individual words, and converted into word embeddings using the pre-trained Word2Vec model from Google.

# Usage
In the provided Python script, there are two functions, run_code_for_training and run_code_for_testing, that respectively train and evaluate the custom GRU model. During training, the model's training loss is tracked and recorded at regular intervals (specified by the variable i). The loss records are then plotted as a graph with iterations on the x-axis and the loss value on the y-axis. This graph can be found by running the script and looking for the file named customGRU_train_loss.jpg.

During testing, the model's accuracy is computed and the confusion matrix is generated using the provided testing dataset. The accuracy is the percentage of the samples that the model correctly classified, and the confusion matrix shows how many samples were classified correctly and how many were classified incorrectly. The function run_code_for_testing outputs both the accuracy and the confusion matrix, and plotconfusionmatrix can be used to generate a heatmap visualization of the confusion matrix. The accuracy and the heatmap can be found by running the script and looking for the files named customGRU_accuracy.jpg and confusion_matrix_customGRU.jpg, respectively.

These metrics are useful for evaluating the performance of the model and determining whether it is performing well or not. The confusion matrix can provide insight into which classes the model is struggling with and help with identifying potential improvements.

# Custom Implementation of Gated Recurrent Units (customGRU)
In this implementation of the Gated Recurrent Units (GRUs), the customGRU class is used to implement a custom version of the GRU logic. The forward function is defined to take in an input tensor data and a hidden state tensor hidden, with an optional boolean parameter sequence_end. The input tensor data represents a single tokenized review, and the hidden state tensor hidden represents the hidden state of the model from the previous time step. The boolean sequence_end flag is used to indicate whether the current input tensor data is the last in the sequence.

The custom GRU implementation uses three separate sequential layers to compute the forget gate, the output from the hidden layer, and the output of the GRU cell. The forget gate is used to decide how much of the previous hidden state to retain, while the output layer produces the current hidden state of the GRU cell. At the final time step, the forward function returns the final output tensor final_out, which is passed through the final sequential layer to produce the output tensor of the GRU cell. The init_hidden function is used to initialize the hidden state tensor hidden with zero values, which is the standard method for initializing RNNs.

# Dependencies
- Pytorch
- NumPy
- Gensim
- sklearn
- matplotlib
- seaborn

# References
- Pytorch
- Gensim
- GoogleNews-vectors-negative300.bin
