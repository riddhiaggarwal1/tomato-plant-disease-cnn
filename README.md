# Tomato Leaf Disease Classification with a CNN

Image classifier that recognises 10 classes of tomato leaf (diseases plus healthy), built in Google Colab for my MSc Data Analytics (BSBI, Berlin).

## What's inside
- `tomato_disease_cnn.ipynb`: the full notebook, with outputs

## What I did
- Loaded and rescaled leaf images (224×224) with a train/validation split using `ImageDataGenerator`
- Built a convolutional neural network in Keras (Conv2D, MaxPooling, BatchNormalization, Dropout, Dense layers) and trained it for 30 epochs
- Evaluated it with accuracy, a confusion matrix, precision, recall, F1 and ROC curves per class

## How to run
Open it in Google Colab, add the tomato leaf image dataset (train/validation folders), and run the cells in order.

**Tools:** Python, TensorFlow/Keras, scikit-learn, Matplotlib, Seaborn, Google Colab
