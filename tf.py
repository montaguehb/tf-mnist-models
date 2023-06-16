"""_summary_
  classify grayscale images of handwritten digits between 0 and 9
  each image is 28x28 pixels, data is provided by MNIST with 60k training images and 10k test images
   
"""

import tensorflow as tf

mnist = tf.keras.datasets.mnist

#separating the data provided by the load_data method into their respective variables
#we divide the x data by 255.0 to convert the data points to a float between 0 and 1
#x refers to the images and y to the labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))
