import tensorflow as tf
import readDatabase
import os
import numpy as np

class ModelArchitecture(tf.keras.Model):
    def __init__(self):
        # super() usecases: The super() builtin returns a proxy object (temporary object of the superclass) that allows us to access methods of the base class. In Python, super() has two major use cases:
        # Allows us to avoid using the base class name explicitly
        # Working with Multiple Inheritance  https://www.programiz.com/python-programming/methods/built-in/super
        
        super(ModelArchitecture, self).__init__()
        # model from paper
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2,strides=2)
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(100, activation='relu')
        self.d2 = tf.keras.layers.Dense(84, activation='relu')
        self.d3 = tf.keras.layers.Dense(10)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,28, 28,1), dtype=tf.float32)])
    def call(self,x, mask=None):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)   
        return x

# Compiles a function into a callable TensorFlow graph.
class ModelTraining(ModelArchitecture):
    def __init__(self):
        super(ModelTraining, self).__init__()
        self.model = ModelArchitecture()
        self.lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.trainLoss = tf.keras.metrics.Mean(name='train_loss')
        self.trainAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.testLoss = tf.keras.metrics.Mean(name='test_loss')
        self.testAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # A decorator takes in a function, adds some functionality and returns it. 
    @tf.function
    def trainingStep(self,data,labels):
        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:
            predictions = self.model(data,training=True)
            loss = self.lossObject(labels,predictions)
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.trainLoss(loss)
        self.trainAccuracy(labels,predictions)

    @tf.function
    def testingStep(self,data,labels):
        predictions = self.model(data,training=False)
        loss = self.lossObject(labels,predictions)
        self.testLoss(loss)
        self.testAccuracy(labels,predictions)

if __name__ == "__main__":
    Epochs = 2
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    [trainData, trainLabels] = readDatabase.readDatabase(path)
    [testData, testLabels] = readDatabase.readDatabase(path, 'test')
    trainData, testData = trainData / 255.0, testData / 255.0
    trainData = trainData[:,:,:,tf.newaxis].astype("float32")
    testData = testData[..., tf.newaxis].astype("float32")

    trainDataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels)).shuffle(10000).batch(32)
    testDataset = tf.data.Dataset.from_tensor_slices((testData, testLabels)).batch(32)

    model = ModelArchitecture()

    modelPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')

    tf.saved_model.save(
    model, modelPath)

    modelTraining = ModelTraining()
    

    for epoch in range(Epochs):
        modelTraining.trainLoss.reset_states()
        modelTraining.trainAccuracy.reset_states()
        modelTraining.testLoss.reset_states()
        modelTraining.testAccuracy.reset_states()

        for trainData, trainLabels in trainDataset:
            modelTraining.trainingStep(trainData, trainLabels)

        for testData, testLabels in testDataset:
            modelTraining.testingStep(testData, testLabels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                            modelTraining.trainLoss.result(),
                            modelTraining.trainAccuracy.result() * 100,
                            modelTraining.testLoss.result(),
                            modelTraining.testAccuracy.result() * 100))

