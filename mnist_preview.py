#Importing tensorflow and matplot lib, a library for plotting various data in python.
import tensorflow as tf
import matplotlib.pyplot as plt

#accessing the dataset
mnist = tf.keras.datasets.mnist

#loading the mnist data into 4 variables split by tuples of training and testing information
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#select the nth training image and print out the label
n = 6
example = train_images[n]
print(train_labels[n])

#Takes in the example image and it's color map
#plt.cm.binary sets the color map to a built in binary color map
#Show displays any open plots
plt.imshow(example, cmap=plt.cm.binary)
plt.show()

