import tensorflow as tf
import random

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 5, input_shape=(100,60), activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(1)
    
    print("Using TensorFlow version " + tf.__version__)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

@tf.function
def train_step(model, loss_object, optimizer, train_loss, train_accuracy, images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  
@tf.function
def test_step(model, loss_object, test_loss, test_accuracy, images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  
def TrainModel(images, labels):
  pct = 0.8
  train_images = []
  train_labels = []
  test_images = []
  test_labels = []
  
  for i in range(len(images)):
    image = images[i]
    label = labels[i]
    if random.random() < pct:
      train_images.append(image)
      train_labels.append(label)
    else:
      test_images.append(image)
      test_labels.append(label)
  print(f'Train samples = {len(train_images)}, test samples = {len(test_images)}')
  
  # x_train, x_test = x_train / 255.0, x_test / 255.0

  # Add a channels dimension
  # x_train = x_train[..., tf.newaxis].astype("float32")
  # x_test = x_test[..., tf.newaxis].astype("float32")

  # train_images.reshape(-1, 100, 60, 1)
  
  train_ds = tf.data.Dataset.from_tensor_slices(
      (train_images, train_labels)).shuffle(10000).batch(32)

  test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

  # Create an instance of the model
  model = MyModel()

  # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam()

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  #train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  #test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

  EPOCHS = 5

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for imgs, lbls in train_ds:
      train_step(model, loss_object, optimizer, train_loss, train_accuracy, imgs, lbls)

    for imgs, lbls in test_ds:
      test_step(model, loss_object, test_loss, test_accuracy, imgs, lbls)

    print(
      f'    Epoch {epoch + 1}, '
      f'Train Loss: {train_loss.result()}, '
      f'Train Accuracy: {train_accuracy.result() * 100}, '
      f'Test Loss: {test_loss.result()}, '
      f'Test Accuracy: {test_accuracy.result() * 100}'
    )
    
  return model

#
# Run the model with the images that don't have a date
#

def RunModel(model, images):
  ages = []
  for image in images:
    age = 12.0     ################################################# TODO: age = model.call(image)
    ages.append(age)
  return ages