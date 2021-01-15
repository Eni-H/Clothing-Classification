import tensorflow as tf
import read_database
import os
import numpy as np
from ModelTraining import*

if __name__ == "__main__":
    Epochs = 3
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    [train_data, train_labels] = read_database.read_database(path)
    [test_data, test_labels] = read_database.read_database(path, 'test')
    train_data, test_data = train_data / 255.0, test_data / 255.0
    train_data = train_data[:,:,:,tf.newaxis].astype("float32")
    test_data = test_data[..., tf.newaxis].astype("float32")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(10000).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    modelArchitecture = ModelArchitecture()
    modelTraining = ModelTraining()

    modelPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')

    tf.saved_model.save(modelArchitecture, modelPath)
    

    for epoch in range(Epochs):
        modelTraining.train_loss.reset_states()
        modelTraining.train_accuracy.reset_states()
        modelTraining.test_loss.reset_states()
        modelTraining.test_accuracy.reset_states()

        for train_data, train_labels in train_dataset:
            modelTraining.training_step(train_data, train_labels)

        for test_data, test_labels in test_dataset:
            modelTraining.testing_step(test_data, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                            modelTraining.train_loss.result(),
                            modelTraining.train_accuracy.result() * 100,
                            modelTraining.test_loss.result(),
                            modelTraining.test_accuracy.result() * 100))

