from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(train_images[0])
# plt.colorbar()
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(0, 25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

#init
model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

#training
history = model.fit(train_images, train_labels, epochs=10)


history_dict = history.history
history_df = pd.DataFrame(history_dict)

#display history
# history_df.plot(figsize=(8,5))
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()


#Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)
predictions = model.predict(test_images)
# predictions[0]
class_names[np.argmax(predictions[0])]

#Display Results:

# define a function that plots the predicted image
def plot_image(i, predictions_array, true_label, img):
    # assign variable names to our parameters
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    # remove grid and axis values 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # display images 
    plt.imshow(img, cmap=plt.cm.binary)
    # return predicted label
    predicted_label = np.argmax(predictions_array)
    # and assign it a colour based on whether it was correct
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    # define label format  
    plt.xlabel("{}{:2.0f}% ({})".format(class_names[predicted_label],
              100*np.max(predictions_array),
              class_names[true_label],
              color=color))

# plot a function to graph the probabilities 

def plot_value_array(i, predictions_array, true_label):
    # assign variable names to our parameters
    predictions_array, true_label = predictions_array[i], true_label[i]
    # remove grid and axis values 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # plot a bar chart
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    # reduce y axis to between 0,1 values 
    plt.ylim([0,1])
    # create prediction 
    predicted_label = np.argmax(predictions_array)
    # set plot colour
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()