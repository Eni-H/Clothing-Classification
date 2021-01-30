import numpy as np
import tensorflow as tf
import os
from read_database import*
from sklearn.metrics import accuracy_score as acc

#visualize evaluation plot confusion
def test_model(model_path,test_path):
    # model = tf.keras.models.load_model(model_path)
    [test_data, test_labels] = read_database(test_path)
    test_data = test_data/255.0
    test_data = test_data[..., tf.newaxis].astype("float32")
    # acc = model.evaluate(test_data, test_labels)

    model = tf.saved_model.load(model_path)
        
    #predict
    predData = np.ndarray(shape=(test_data.shape[0]),dtype = int)
    for i in range(0,test_data.shape[0]):
        y = model.call(test_data[[i],:,:,:])
        predData[i] = np.argmax(y)

    accuracy = acc(test_labels,predData)


    return(accuracy)

test_model('models','data')