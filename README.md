# SymptoMed-Bangkit-Capstone-Project
## DS-ML
The evaluation of the machine learning models in Data Science involved a comprehensive analysis using various methods, including Random Forest, Random Forest using hyperparameters, and Shallow Neural Networks (NN). These models were specifically developed to classify data related to different types of diseases and their related symptoms. The accuracy results obtained from each of these machine-learning models are as follows:

  1. Random Forest: The model achieved an accuracy of 98% in classifying disease types and symptoms. It demonstrated robust performance in capturing complex patterns within the data and making accurate predictions.
  2. Random Forest using hyperparameters: The use of hyperparameters in the Random Forest model did not significantly differ from the regular Random Forest approach. The accuracy of the model remained at an impressive 98%, which is considered highly satisfactory in terms of prediction accuracy.
  3. Shallow NN: The utilization of a Shallow Neural Network yielded an accuracy of 99%. This model leveraged the power of neural networks to learn intricate relationships within the data, enabling more precise disease classification and symptom identification.

The evaluation results indicate that these machine-learning models have proven to be effective in accurately classifying disease types and symptoms. These findings provide valuable insights for further research and potential applications in the medical field, contributing to improved diagnostics and treatment strategies.

## NLP-ML
The NLP model in this application is designed to classify diseases based on user-provided narratives that describe their symptoms. The model is based on transfer learning with BERT Base Indonesian.

The NLP model involves the following steps:

1. Fine-tuning the BERT Base Indonesian model.
2. Various preprocessing steps, including replacing column names, tokenizing, implementing stopwords using Sastrawi.
3. Performing text transformations on the dataset.
4. Label encoding for the target variable.
5. Dataset splitting for training and evaluation.
6. Utilizing a data pipeline for efficient model serving.

The NLP model achieved an evaluation accuracy of 99.1% in classifying diseases based on user narratives.

## Usage
To use the application and leverage the disease classification models, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the application and provide user input narratives for disease classification.






