import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.utils import get_custom_objects


class Swish(KL.Layer):
    """
    Swish activation function layer.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Applies the Swish activation function to the inputs.
        """
        return tf.nn.swish(inputs)


class DropConnect(KL.Layer):
    """
    DropConnect regularization layer.
    
    DropConnect randomly drops connections during training with a specified rate.
    """
    def __init__(self, drop_connect_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):
        """
        Applies DropConnect to the inputs.
        
        Arguments:
        inputs -- Tensor input to the layer.
        training -- Boolean, whether the layer should behave in training mode or inference mode.
        
        Returns:
        Tensor output after applying DropConnect.
        """
        if training:  # This condition will be true during training
            keep_prob = 1.0 - self.drop_connect_rate
            shape = tf.shape(inputs)
            random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            return tf.divide(inputs, keep_prob) * binary_tensor
        else:
            return inputs

    def get_config(self):
        """
        Returns the configuration of the layer as a dictionary.
        """
        config = super().get_config()
        config['drop_connect_rate'] = self.drop_connect_rate
        return config


# Register the custom layers with Keras
get_custom_objects().update({
    'DropConnect': DropConnect,
    'Swish': Swish,
})

