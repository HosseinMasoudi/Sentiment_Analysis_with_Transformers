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

#### stage 1 :
```
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)
output:

Epoch 1/20
2886/2887 [============================>.] - ETA: 0s - loss: 0.5719 - accuracy: 0.6998 - auc: 0.7704 - precision: 0.7201 - recall: 0.6743/usr/local/lib/python3.12/dist-packages/transformers/generation/tf_utils.py:465: UserWarning: `seed_generator` is deprecated and will be removed in a future version.
  warnings.warn("`seed_generator` is deprecated and will be removed in a future version.", UserWarning)
2887/2887 [==============================] - 250s 79ms/step - loss: 0.5719 - accuracy: 0.6998 - auc: 0.7704 - precision: 0.7201 - recall: 0.6743 - val_loss: 0.4716 - val_accuracy: 0.7899 - val_auc: 0.8606 - val_precision: 0.8280 - val_recall: 0.7430 - lr: 1.0000e-05
Epoch 2/20
2887/2887 [==============================] - 210s 73ms/step - loss: 0.4985 - accuracy: 0.7630 - auc: 0.8328 - precision: 0.7963 - recall: 0.7202 - val_loss: 0.4453 - val_accuracy: 0.8007 - val_auc: 0.8706 - val_precision: 0.8388 - val_recall: 0.7548 - lr: 1.0000e-05
Epoch 3/20
2887/2887 [==============================] - 211s 73ms/step - loss: 0.4791 - accuracy: 0.7756 - auc: 0.8465 - precision: 0.8091 - recall: 0.7337 - val_loss: 0.4347 - val_accuracy: 0.8034 - val_auc: 0.8753 - val_precision: 0.8343 - val_recall: 0.7673 - lr: 1.0000e-05
Epoch 4/20
2887/2887 [==============================] - 213s 74ms/step - loss: 0.4692 - accuracy: 0.7824 - auc: 0.8524 - precision: 0.8166 - recall: 0.7401 - val_loss: 0.4289 - val_accuracy: 0.8082 - val_auc: 0.8786 - val_precision: 0.8516 - val_recall: 0.7562 - lr: 1.0000e-05
Epoch 5/20
2887/2887 [==============================] - 233s 81ms/step - loss: 0.4648 - accuracy: 0.7840 - auc: 0.8555 - precision: 0.8173 - recall: 0.7433 - val_loss: 0.4247 - val_accuracy: 0.8092 - val_auc: 0.8808 - val_precision: 0.8538 - val_recall: 0.7558 - lr: 1.0000e-05
Epoch 6/20
2887/2887 [==============================] - 214s 74ms/step - loss: 0.4608 - accuracy: 0.7851 - auc: 0.8578 - precision: 0.8180 - recall: 0.7450 - val_loss: 0.4211 - val_accuracy: 0.8108 - val_auc: 0.8826 - val_precision: 0.8531 - val_recall: 0.7606 - lr: 1.0000e-05
Epoch 7/20
2887/2887 [==============================] - 240s 83ms/step - loss: 0.4564 - accuracy: 0.7889 - auc: 0.8607 - precision: 0.8240 - recall: 0.7459 - val_loss: 0.4184 - val_accuracy: 0.8118 - val_auc: 0.8842 - val_precision: 0.8578 - val_recall: 0.7570 - lr: 1.0000e-05
Epoch 8/20
2887/2887 [==============================] - 219s 76ms/step - loss: 0.4533 - accuracy: 0.7905 - auc: 0.8624 - precision: 0.8252 - recall: 0.7483 - val_loss: 0.4180 - val_accuracy: 0.8134 - val_auc: 0.8857 - val_precision: 0.8714 - val_recall: 0.7444 - lr: 1.0000e-05
Epoch 9/20
2887/2887 [==============================] - 230s 80ms/step - loss: 0.4480 - accuracy: 0.7943 - auc: 0.8656 - precision: 0.8307 - recall: 0.7501 - val_loss: 0.4134 - val_accuracy: 0.8143 - val_auc: 0.8864 - val_precision: 0.8455 - val_recall: 0.7786 - lr: 1.0000e-05
Epoch 10/20
2887/2887 [==============================] - 236s 82ms/step - loss: 0.4486 - accuracy: 0.7927 - auc: 0.8651 - precision: 0.8290 - recall: 0.7484 - val_loss: 0.4120 - val_accuracy: 0.8155 - val_auc: 0.8876 - val_precision: 0.8598 - val_recall: 0.7632 - lr: 1.0000e-05
Epoch 11/20
2887/2887 [==============================] - 234s 81ms/step - loss: 0.4443 - accuracy: 0.7970 - auc: 0.8677 - precision: 0.8341 - recall: 0.7522 - val_loss: 0.4099 - val_accuracy: 0.8163 - val_auc: 0.8884 - val_precision: 0.8548 - val_recall: 0.7713 - lr: 1.0000e-05
Epoch 12/20
2887/2887 [==============================] - 217s 75ms/step - loss: 0.4440 - accuracy: 0.7955 - auc: 0.8677 - precision: 0.8324 - recall: 0.7509 - val_loss: 0.4093 - val_accuracy: 0.8173 - val_auc: 0.8894 - val_precision: 0.8655 - val_recall: 0.7604 - lr: 1.0000e-05
Epoch 13/20
2887/2887 [==============================] - 220s 76ms/step - loss: 0.4420 - accuracy: 0.7966 - auc: 0.8691 - precision: 0.8337 - recall: 0.7516 - val_loss: 0.4079 - val_accuracy: 0.8170 - val_auc: 0.8901 - val_precision: 0.8633 - val_recall: 0.7624 - lr: 1.0000e-05
Epoch 14/20
2887/2887 [==============================] - 218s 75ms/step - loss: 0.4413 - accuracy: 0.7956 - auc: 0.8694 - precision: 0.8344 - recall: 0.7483 - val_loss: 0.4061 - val_accuracy: 0.8189 - val_auc: 0.8910 - val_precision: 0.8655 - val_recall: 0.7641 - lr: 1.0000e-05
Epoch 15/20
2887/2887 [==============================] - 213s 74ms/step - loss: 0.4370 - accuracy: 0.7994 - auc: 0.8721 - precision: 0.8373 - recall: 0.7538 - val_loss: 0.4057 - val_accuracy: 0.8200 - val_auc: 0.8915 - val_precision: 0.8720 - val_recall: 0.7590 - lr: 1.0000e-05
Epoch 16/20
2887/2887 [==============================] - 224s 77ms/step - loss: 0.4385 - accuracy: 0.7978 - auc: 0.8711 - precision: 0.8384 - recall: 0.7484 - val_loss: 0.4038 - val_accuracy: 0.8195 - val_auc: 0.8922 - val_precision: 0.8669 - val_recall: 0.7639 - lr: 1.0000e-05
Epoch 17/20
2887/2887 [==============================] - 213s 74ms/step - loss: 0.4353 - accuracy: 0.8020 - auc: 0.8729 - precision: 0.8408 - recall: 0.7552 - val_loss: 0.4026 - val_accuracy: 0.8200 - val_auc: 0.8928 - val_precision: 0.8665 - val_recall: 0.7655 - lr: 1.0000e-05
Epoch 18/20
2887/2887 [==============================] - 239s 83ms/step - loss: 0.4344 - accuracy: 0.8003 - auc: 0.8734 - precision: 0.8398 - recall: 0.7525 - val_loss: 0.4017 - val_accuracy: 0.8200 - val_auc: 0.8935 - val_precision: 0.8673 - val_recall: 0.7645 - lr: 1.0000e-05
Epoch 19/20
2887/2887 [==============================] - 221s 77ms/step - loss: 0.4326 - accuracy: 0.8026 - auc: 0.8748 - precision: 0.8427 - recall: 0.7542 - val_loss: 0.4010 - val_accuracy: 0.8212 - val_auc: 0.8938 - val_precision: 0.8692 - val_recall: 0.7651 - lr: 1.0000e-05
Epoch 20/20
2887/2887 [==============================] - 220s 76ms/step - loss: 0.4339 - accuracy: 0.8010 - auc: 0.8738 - precision: 0.8413 - recall: 0.7521 - val_loss: 0.3998 - val_accuracy: 0.8211 - val_auc: 0.8940 - val_precision: 0.8560 - val_recall: 0.7812 - lr: 1.0000e-05

```
#### stage 2 :
```
history_2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)
output: 

2887/2887 [==============================] - 561s 182ms/step - loss: 0.4142 - accuracy: 0.8131 - auc: 0.8864 - precision: 0.8504 - recall: 0.7694 - val_loss: 0.3380 - val_accuracy: 0.8563 - val_auc: 0.9260 - val_precision: 0.8979 - val_recall: 0.8108 - lr: 5.0000e-06
Epoch 2/5
2887/2887 [==============================] - 454s 157ms/step - loss: 0.3491 - accuracy: 0.8502 - auc: 0.9197 - precision: 0.8920 - recall: 0.8041 - val_loss: 0.3336 - val_accuracy: 0.8625 - val_auc: 0.9320 - val_precision: 0.9271 - val_recall: 0.7930 - lr: 5.0000e-06
Epoch 3/5
2887/2887 [==============================] - 476s 165ms/step - loss: 0.3256 - accuracy: 0.8625 - auc: 0.9305 - precision: 0.9015 - recall: 0.8203 - val_loss: 0.3169 - val_accuracy: 0.8666 - val_auc: 0.9357 - val_precision: 0.8896 - val_recall: 0.8435 - lr: 5.0000e-06
Epoch 4/5
2887/2887 [==============================] - 464s 161ms/step - loss: 0.3110 - accuracy: 0.8697 - auc: 0.9370 - precision: 0.9074 - recall: 0.8295 - val_loss: 0.3125 - val_accuracy: 0.8697 - val_auc: 0.9376 - val_precision: 0.9142 - val_recall: 0.8221 - lr: 5.0000e-06
Epoch 5/5
2887/2887 [==============================] - 415s 144ms/step - loss: 0.2965 - accuracy: 0.8770 - auc: 0.9428 - precision: 0.9121 - recall: 0.8401 - val_loss: 0.3180 - val_accuracy: 0.8667 - val_auc: 0.9375 - val_precision: 0.8957 - val_recall: 0.8364 - lr: 5.0000e-06
```
From the model output, it can be concluded that the model is overfitting after epoch 3

### Adam vs AdamW

AdamW is a variant of the Adam optimizer that separates weight decay from the gradient update based on the observation that the weight decay formulation is different when applied to SGD and Adam.

When to Choose Adam: You can use Adam for quick prototyping or simpler tasks where regularization is not crucial. It may converge faster initially but can suffer from poor generalization due to interference from weight decay.

When to Choose AdamW: In case you have larger models or when training on complex, high-dimensional data, it’s better to choose AdamW, because the decoupled weight decay helps achieve better generalization and stable convergence.

### UI creation:


The last part of the project, which is related to creating a front for the model, we create a page with the help of the streamlit library for the user to enter their text input.
The text input is processed before sending to model 2:


1. Normalization
2. Spell correction

### Spelling correction:


Since there is a possibility of spelling errors in the sent data, we solved this problem by sending an API call

Basically, we send the text to the Spelling site and this site returns the correct word to us if there is an error. If the word sent by the user is confirmed, it replaces the original word.
