## Sentiment Analysis Project



Sentiment Analysis Project (Two-Class) Using Fine-Tuning the ParsBERT Model

### Dataset:


This dataset is related to Snap comments, which are initially used to collect student feedback on instructor evaluations.

### Hazm Library Problem:


This is where some of the Hazm prerequisites do not match the TensorFlow prerequisites. In essence, this is why we use Environment\
so that if someone else wants to run the model, the required requirements are automatically installed and the model cannot be run with problems.

Hazm is not compatible with numpy>=1.26.4, but it can be installed in Linux and MacOs environments, and given that MacO processors are suitable\
for Deep Learning processing and we need a GPU to train the model, I started writing the normalizer function.

### The normalizer function (JackageNormalizer:


consists of 9 parts that perform the normalization process in order:
```
.Normalize_unicode
.Remove_unwanted_characters
.Convert_numbers
.Convert_numbers_to_words
.Standardize_persian_text
.Remove_keshide
.Remove_punctuation
.Fix_persian_zwnj
.Remove_stopwords
```

### Model execution:
In order to solve the normalization problem, we ran the model on the GoogleColab T4 GPU
Then we start preprocessing the unclean data of the dataset to prepare it for training the model.

The result of the model training is as follows:

```
Epoch 1/3
2887/2887 [==============================] - 874s 282ms/step - loss: 0.3201 - sparse_categorical_accuracy: 0.8674 - val_loss: 0.2871 - val_sparse_categorical_accuracy: 0.8874

Epoch 2/3
2887/2887 [==============================] - 803s 278ms/step - loss: 0.2543 - sparse_categorical_accuracy: 0.8973 - val_loss: 0.2988 - val_sparse_categorical_accuracy: 0.8803

Epoch 3/3
2887/2887 [==============================] - 802s 278ms/step - loss: 0.1908 - sparse_categorical_accuracy: 0.9267 - val_loss: 0.3572 - val_sparse_categorical_accuracy: 0.8605


 Final Evaluation on Test Set
619/619 [==============================] - 57s 87ms/step - loss: 0.3776 - sparse_categorical_accuracy: 0.8516
[0.3776339590549469, 0.8515561819076538]
```
From the model output, it can be concluded that the model is overfitting after epoch 2

### UI creation:


The last part of the project, which is related to creating a front for the model, we create a page with the help of the streamlit library for the user to enter their text input.
The text input is processed before sending to model 2:


1. Normalization
2. Spell correction

### Spelling correction:


Since there is a possibility of spelling errors in the sent data, we solved this problem by sending an API call

Basically, we send the text to the Spelling site and this site returns the correct word to us if there is an error. If the word sent by the user is confirmed, it replaces the original word.
