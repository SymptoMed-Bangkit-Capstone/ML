# SymptoMed-Bangkit-Capstone-Project

In this application, the Machine Learning team has built two models for disease classification. These include the Data Science and Natural Language Processing (NLP) models. The Data Science model is used when the application is in offline mode, while the NLP model is used when the application is in online mode.

The Data Science model utilizes a Shallow Neural Network implemented with TensorFlow. This model achieves an accuracy of 100% in classifying the 41 diseases.

The NLP model is based on transfer learning with BERT Base Indonesian. It involves fine-tuning the model and various preprocessing steps such as replacing column names, tokenizing, implementing stopwords using Sastrawi, performing text transformations on the dataset, label encoding, dataset splitting, and utilizing a data pipeline for model serving. The NLP model achieved an evaluation accuracy of 99.1%.
