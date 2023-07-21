"""_summary_
  classify grayscale images of handwritten digits between 0 and 9
  each image is 28x28 pixels, data is provided by MNIST with 60k training images and 10k test images
   
"""

import tensorflow as tf

mnist = tf.keras.datasets.mnist

#separating the data provided by the load_data method into their respective variables
#we divide the x data by 255.0 to convert the data points to a float between 0 and 1
#x refers to the images and y to the labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(train_images[:1]).numpy()
tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(train_labels[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images,  test_labels, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(test_images[:5]))
