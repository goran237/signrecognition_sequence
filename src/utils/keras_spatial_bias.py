import keras.backend as kb
import tensorflow as tf
from keras.layers import Layer

def _kb_linspace(num):
    num = kb.cast(num, kb.floatx())
    return kb.arange(0, num, dtype=kb.floatx()) / (num - 1)

def _kb_grid_coords(width, height):
    w, h = width, height
        
    coords = kb.stack(
        [
            kb.reshape(kb.tile(kb.expand_dims(_kb_linspace(num=w), -1), [1, h]), [-1]),
            kb.tile(_kb_linspace(num=h), [w]),
        ],
        axis=-1,
    )
    coords = kb.reshape(coords, [w, h, 2])
    return coords

class ConcatSpatialCoordinate(Layer):

    def __init__(self, **kwargs):
        """Concatenates the (x, y) coordinate normalised to 0-1 to each spatial location in the image.
        Allows a network to learn spatial bias. Has been explored in at least one paper,
        "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution"
        https://arxiv.org/abs/1807.03247
        
        Improves performance where spatial bias is appropriate.
        
        Works with dynamic shapes.
        
        # Example
        ```python
        x_input = Input([None, None, 1])
        x = ConcatSpatialCoordinate()(x_input)

        model = Model(x_input, x)

        output = model.predict(np.zeros([1, 3, 3, 1]))
        spatial_features = output[0, :, :, -2:]
        assert np.all(spatial_features[0,   0] == [0, 0])
        assert np.all(spatial_features[-1, -1] == [1, 1])
        assert np.all(spatial_features[0,  -1] == [0, 1])

        # Because this example was 3x3, cordinates are [0, 0.5, 1], so
        assert np.all(spatial_features[1,  1] == [0.5, 0.5])
        ```
        """

        
        if kb.image_data_format() != 'channels_last':
            raise Exception((
                "Only compatible with" 
                " kb.image_data_format() == 'channels_last'"))
    
        super(ConcatSpatialCoordinate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatSpatialCoordinate, self).build(input_shape) 

    def call(self, x):
        dynamic_input_shape = kb.shape(x)
        w = dynamic_input_shape[-3]
        h = dynamic_input_shape[-2]
        nb_batch = dynamic_input_shape[0]
        bias = _kb_grid_coords(width=w, height=h)
        return kb.concatenate([x, tf.tile(kb.expand_dims(bias, 0), [nb_batch, 1, 1, 1])], axis=-1)

    def compute_output_shape(self, input_shape):
        batch_size, w, h, channels = input_shape
        return (batch_size, w, h, channels + 2)
    
    
def test_ConcatSpatialCoordinate():
    import numpy as np
    from keras.layers import Input
    from keras.models import Model

    x_input = Input([None, None, 1])
    x = ConcatSpatialCoordinate()(x_input)

    model = Model(x_input, x)

    output = model.predict(np.zeros([1, 3, 3, 1]))
    spatial_features = output[0, :, :, -2:]
    
    # The following are always true.
    assert np.all(spatial_features[0,   0] == [0, 0])
    assert np.all(spatial_features[-1, -1] == [1, 1])
    assert np.all(spatial_features[0,  -1] == [0, 1])

    # Because this example was 3x3, cordinates are [0, 0.5, 1], so
    assert np.all(spatial_features[1,  1] == [0.5, 0.5])
    
    
if __name__ == '__main__':
    test_ConcatSpatialCoordinate()