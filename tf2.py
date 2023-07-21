import tensorflow as tf

mnist = tf.keras.datasets.mnist

#load the mnist data into their respective variables, divide by 255.0 to get each data point to be between 0 and 1
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# 2 fully connected layers with the last being a 10-way softmax classification layer
# First layer has an output size of 512 values
# returns 10 scores betweeon 0 and 1 with each representing the likely hood of a given number

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, "relu"),
    tf.keras.layers.Dense(10, "softmax")
])

model.compile(optimizer="rmsprop",
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

predictions = model.predict(test_images[:1])
print(test_labels[0])
print(predictions[0])

test_loss, test_acc = model.evaluate(test_images, test_labels)