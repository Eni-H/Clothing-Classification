# turn a .pb model to .tflite model;
# if wanted check the accurracy of .tflite model;  
#imports
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score as acc
from sklearn import metrics
from read_database import*

class QuantizeModel():
    def __init__(self,model_path,data_path=None):
        self.model_path = model_path
        self.data_path = data_path
    #measure the dynamic range of activations and inputs
    def representative_dataset(self):
        [train_data, train_labels] = read_database(self.data_path)
        train_data = train_data/255.0
        train_data = train_data[:,:,:,tf.newaxis].astype("float32")
        for data in tf.data.Dataset.from_tensor_slices((train_data)).batch(1).take(100):
            yield [data]

    def quantize_model(self,quantization_type=None):
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        
        if quantization_type == 'dynamic_range':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]    
        elif quantization_type == 'int_with_float_fallback':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset
        elif quantization_type == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8  # or tf.uint8
            converter.inference_output_type = tf.int8  # or tf.uint8
        elif quantization_type == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == 'int16':
            converter.representative_dataset = self.representative_dataset
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        tflite_model = converter.convert()
        #write model
        open(os.path.join(self.model_path,'model.tflite'),'wb').write(tflite_model)

    def test_converter(self):
        interpreter = tf.lite.Interpreter(os.path.join(self.model_path,'model.tflite'))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        #load test data
        [test_data, test_labels] = read_database(self.data_path)
        test_data = test_data/255.0
        test_data = test_data[..., tf.newaxis].astype("float32")
        predData = np.ndarray(shape=(test_data.shape[0]),dtype = 'uint8')

        for i in range(0,test_data.shape[0]):
            test_data_temp = np.array(test_data[[i],:,:,:],dtype='float32')
            interpreter.set_tensor(input_details[0]['index'],test_data_temp)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predData[i] = np.argmax(output_data)

        kappa=metrics.cohen_kappa_score(test_labels,predData)
        print('Kappa:',kappa)
        accuracy = acc(test_labels,predData)
        print('Accuracy:',accuracy)
        print('Confusion matrix:')
        confusion_mat = metrics.confusion_matrix(test_labels,predData)
        print(confusion_mat)

        return(accuracy)


if __name__ == "__main__":
    
    quantization_type = 'int8'
    quantizeModel = QuantizeModel('models','data')
    quantizeModel.quantize_model(quantization_type)
    quantizeModel.test_converter()
    
