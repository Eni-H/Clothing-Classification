from model_architecture import*

# Compiles a function into a callable TensorFlow graph.
class ModelTraining(ModelArchitecture):
    def __init__(self):
        super(ModelTraining, self).__init__()
        self.modelArchitecture = ModelArchitecture()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # A decorator takes in a function, adds some functionality and returns it. 
    @tf.function
    def training_step(self,data,labels):
        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:
            predictions = self.modelArchitecture(data,training=True)
            loss = self.loss_object(labels,predictions)
        gradients = tape.gradient(loss,self.modelArchitecture.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.modelArchitecture.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels,predictions)

    @tf.function
    def testing_step(self,data,labels):
        predictions = self.modelArchitecture(data,training=False)
        loss = self.loss_object(labels,predictions)
        self.test_loss(loss)
        self.test_accuracy(labels,predictions)