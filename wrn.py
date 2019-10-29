from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D, Input, Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add

from keras_applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

def WideResidualNetwork(depth = 28, width = 8, dropout_rate = 0.0,
                        include_top = True, weights = None,
                        input_tensor = None, input_shape = None,
                        classes = 10, activation = 'softmax'):

    '''
    Instatiate a Wide Residual Network
    '''
    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4) should be divisible by 6')

    # Determine proper input input_shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size = 32,
                                      min_size = 8,
                                      data_format = K.image_dim_ordering(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_wide_residual_network(classes, img_input, include_top, depth,
                                       width, dropout_rate, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='wide-resnet')
    return model


def __conv1_block(input):
    x = Conv2D(16, (3,3), padding = 'same')(input)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)
    return x

def __conv2_block(input, k = 1, dropout = 0.0):
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # check if input number of filters is same as 16 * k, else create conv2d for this input_tensor
    if K.image_data_format() == 'channels_first':
        if init._shape_val[1] != 16 * k:
            init = Conv2D(16*k, (1,1), activation='linear', padding = 'same')(init)
    else:
        if init._shape_val[-1] != 16 * k:
            init = Conv2D(16*k, (1,1), activation='linear', padding = 'same')(init)

    x = Conv2D(16 * k, (3,3), padding = 'same')(input)
    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x]) # it appears this is where the residual merge is
    return m

def __conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 32 * k, else
    # create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._shape_val[1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init._shape_val[-1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(32 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(32 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m

def ___conv4_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == 'th' else -1

    # Check if input number of filters is same as 64 * k, else
    # create convolution2d for this input
    if K.image_dim_ordering() == 'th':
        if init._shape_val[1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)
    else:
        if init._shape_val[-1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(64 * k, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(64 * k, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = Add()([init, x])
    return m

def __create_wide_residual_network(nb_classes, img_input, include_top, depth = 28,
                                   width = 8, dropout = 0.0, activation = 'softmax'):

    '''
    This is not the primary function call...

    nb_classes (Int): output shape
    img_input ()???
    '''

    N = (depth - 4) // 6

    x = __conv1_block(img_input)
    nb_conv = 4

    for i in range(N):
        x = __conv2_block(x, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = __conv3_block(x, width, dropout)
        nb_conv += 2

    x = MaxPooling2D((2, 2))(x)

    for i in range(N):
        x = ___conv4_block(x, width, dropout)
        nb_conv += 2

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation=activation)(x)

    return x
