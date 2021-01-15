import tensorflow as tf

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