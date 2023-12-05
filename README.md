# Project Name: Click Prediction Model

## Project Description

This project aims to develop a machine learning (ML) model to predict the probability of a click based on the provided dataset. The dataset includes various features related to user interactions and device information. The primary tasks involve feature analysis, model selection, hyperparameter tuning, and model validation.

## Table of Contents

- [Data](#data)
- [Data Processing](#data-processing)
- [Training](#training)

## Data

The dataset consists of the following columns:

- `reg_time`: Registration timestamp
- `uid`: User ID
- `fc_imp_chk`: Feature indicating some check
- `fc_time_chk`: Another feature indicating time check
- `utmtr`: UTM Tracker
- `mm_dma`: Multimedia DMA
- `osName`: Operating System Name
- `model`: Device model
- `hardware`: Hardware information
- `site_id`: Site ID

## Data Processing

The data processing notebook includes the following steps:

1. Loading the dataset from CSV files (`interview.X.csv` and `interview.y.csv`).
2. Removing duplicate entries based on the `uid` column.
3. Standardizing the `hardware` column by replacing "Mobile+Phone" with "Mobile Phone".
4. Creating dictionaries (`d_model`, `d_os`, and `d_hardware`) to store most repeated values for imputation.
5. Imputing missing values in the `osName`, `model`, and `hardware` columns based on the created dictionaries, so NaN counts decreased from $24472$ to $23667$.
6. Converting the `reg_time` column to a numerical representation and normalizing other numerical columns.
7. Encoding categorical columns using one-hot encoding.
8. Obtaining word embeddings for the `site_id` column using OpenAI's text-embedding model.
9. Saving the processed data as `data_norm_X.npy` and `data_norm_Y.npy`.

## Training

The training notebook includes the following steps:

1. Loading the preprocessed data from `data_norm_X.npy` and `data_norm_Y.npy`.
2. Splitting the data into training, validation, and test sets.
3. Creating a custom PyTorch dataset class (`CustomDataset`).
4. Defining the neural network model (`Model`) with four linear layers and batch normalization.
5. Training the model using the Adam optimizer and negative log-likelihood loss.
    Loss Function

    The loss function is defined as:

    ${Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\max\left(\hat{y}_{i, \text{label}}, \epsilon\right)\right) $

    where $\hat{y}_{i, \text{label}}$ represents the predicted probability for the correct class of example $i$, and $\epsilon$ is a small constant to avoid numerical instability.
6. Saving the best model based on validation loss.
7. Evaluating the model on the test set and calculating precision, recall, and accuracy.
8. Visualizing the confusion matrix and Integrated Gradients attribution for feature importance.
    
    [Integrated Gradients](https://medium.com/@kemalpiro/xai-methods-integrated-gradients-6ee1fe4120d8) is an interpretability technique used to explain the model's predictions. Given an input example $x$ and a target class label $c$, the attribution of the $i$-th feature ($x_i$) is computed as follows:

    ${Attribution}(x_i) = (x_i - x_i') \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x_i} \, d\alpha $

    where $F$ is the model's prediction function, $x'$ is a baseline input, and $\frac{\partial F(x)}{\partial x_i}$ is the partial derivative of the model's prediction with respect to feature $i$. The integral is approximated using numerical methods.