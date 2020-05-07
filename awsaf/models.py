# Network 2

def Network2(Input_shape):

  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate
  from keras.layers import MaxPool2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop
  
  xin = Input(Input_shape)

  x = Conv2D(16, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x1a = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(16, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Conv2D(16, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)



  x1a = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)  
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Conv2D(32, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)


  x1a = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Dropout(0.5)(x)
  x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)

  x1a = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Dropout(0.5)(x)
  x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)

  x1a = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
  x   = BatchNormalization()(x1a)

  x1b = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)

  x1d = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1c)
  x1d = BatchNormalization()(x1d)


  x = Concatenate(axis=-1)([x1a, x1b, x1c, x1d])
  x = Dropout(0.5)(x)
  x = Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = MaxPool2D()(x)

  x1a = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
  x1a = BatchNormalization()(x1a)

  x1b = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1a)
  x1b = BatchNormalization()(x1b)

  x1c = SeparableConv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x1b)
  x1c = BatchNormalization()(x1c)


  x = Concatenate(axis=-1)([x1a, x1b, x1c])
  x = Dropout(0.5)(x)
  x = Conv2D(256, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)

  x = GlobalAveragePooling2D()(x)

  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)

  out = Dense(1, activation='sigmoid')(x)

  model = Model(xin, out)
  
  model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
  return model
  
  
  
  
  
# CovXNet :

## Residual Unit:

def Residual_Unit(input_tensor, nb_of_input_channels, max_dilation, number_of_units):

  for i in range(number_of_units):
    x1 = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
    x1 = BatchNormalization()(x1)

    a = []

    for i in range(1, max_dilation+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)
    x = Conv2D(nb_of_input_channels, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])

    input_tensor = x

  return x
  
  
## Shifter Unit:

def Shifter_Unit(input_tensor, nb_of_input_channels, max_dilation):

    x1 = Conv2D(nb_of_input_channels*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
    x1 = BatchNormalization()(x1)

    a = []

    for i in range(1, max_dilation+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = MaxPool2D(pool_size=(2,2))(temp)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)

    x = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    return x
    

## CovXNet256:

def CovXNet256(input_shape, nb_class, depth):
  
  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate
  from keras.layers import MaxPool2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop
  
  def Residual_Unit(input_tensor, nb_of_input_channels, max_dilation, number_of_units):

    for i in range(number_of_units):
      x1 = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
      x1 = BatchNormalization()(x1)

      a = []

      for i in range(1, max_dilation+1):
        temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
        temp = BatchNormalization()(temp)
        a.append(temp)

      x = Concatenate(axis= -1)(a)
      x = Conv2D(nb_of_input_channels, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
      x = BatchNormalization()(x)

      x = Add()([x, input_tensor])

      input_tensor = x

    return x
  
  def Shifter_Unit(input_tensor, nb_of_input_channels, max_dilation):

    x1 = Conv2D(nb_of_input_channels*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
    x1 = BatchNormalization()(x1)

    a = []

    for i in range(1, max_dilation+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = MaxPool2D(pool_size=(2,2))(temp)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)

    x = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    return x
  
  
  xin = Input(shape= input_shape)

  x = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x = Conv2D(32, kernel_size = (3,3), strides= (2,2), padding = 'same', activation='relu')(x)
  x = BatchNormalization()(x)
  
##Max Dilation rate will be vary in the range (1,6). 

# Max Dilation rate is 6 for tensor (128x128x32)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=6, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=6)


# Max Dilation rate is 5 for (64x64x64)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=5, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=5)

# Max Dilation rate is 4 for (32x32x128)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=4, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=4)

# Max Dilation rate is 3 for (16x16x128)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=256, max_dilation=3, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=256, max_dilation=3)

# Max Dilation rate is 2 for (8x8x256)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=512, max_dilation=2, number_of_units=depth)

  x = GlobalAveragePooling2D()(x)

  x = Dense(64, activation='relu')(x)
  x = Dense(nb_class, activation= 'softmax')(x)

  model = Model(xin, x)

  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  return model
 
 
## CovXNet128: 

def CovXNet128(input_shape, nb_class, depth):
  
  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate
  from keras.layers import MaxPool2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop
  
  def Residual_Unit(input_tensor, nb_of_input_channels, max_dilation, number_of_units):

    for i in range(number_of_units):
      x1 = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
      x1 = BatchNormalization()(x1)

      a = []

      for i in range(1, max_dilation+1):
        temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
        temp = BatchNormalization()(temp)
        a.append(temp)

      x = Concatenate(axis= -1)(a)
      x = Conv2D(nb_of_input_channels, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
      x = BatchNormalization()(x)

      x = Add()([x, input_tensor])

      input_tensor = x

    return x
  
  def Shifter_Unit(input_tensor, nb_of_input_channels, max_dilation):

    x1 = Conv2D(nb_of_input_channels*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
    x1 = BatchNormalization()(x1)

    a = []

    for i in range(1, max_dilation+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = MaxPool2D(pool_size=(2,2))(temp)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)

    x = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    return x
  

  xin = Input(shape= input_shape)

  x = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x = Conv2D(32, kernel_size = (3,3), strides= (2,2), padding = 'same', activation='relu')(x)
  x = BatchNormalization()(x)
  
##Max Dilation rate will be vary in the range (1,5). 

# Max Dilation rate is 5 for tensor (64x64x32)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=5, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=5)


# Max Dilation rate is 4 for (32x32x64)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=4, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=4)

# Max Dilation rate is 3 for (16x16x128)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=3, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=3)

# Max Dilation rate is 2 for (8x8x256)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=256, max_dilation=2, number_of_units=depth)

  x = GlobalAveragePooling2D()(x)

  x = Dense(64, activation='relu')(x)
  x = Dense(nb_class, activation= 'softmax')(x)

  model = Model(xin, x)

  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  return model
  
  
  
  
## CovXNet64:
 
def CovXNet64(input_shape, nb_class, depth):
  
  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate
  from keras.layers import MaxPool2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop
  
  def Residual_Unit(input_tensor, nb_of_input_channels, max_dilation, number_of_units):

    for i in range(number_of_units):
      x1 = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
      x1 = BatchNormalization()(x1)

      a = []

      for i in range(1, max_dilation+1):
        temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
        temp = BatchNormalization()(temp)
        a.append(temp)

      x = Concatenate(axis= -1)(a)
      x = Conv2D(nb_of_input_channels, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
      x = BatchNormalization()(x)

      x = Add()([x, input_tensor])

      input_tensor = x

    return x
  
  def Shifter_Unit(input_tensor, nb_of_input_channels, max_dilation):

    x1 = Conv2D(nb_of_input_channels*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
    x1 = BatchNormalization()(x1)

    a = []

    for i in range(1, max_dilation+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = MaxPool2D(pool_size=(2,2))(temp)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)

    x = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    return x
  
 
  xin = Input(shape= input_shape)

  x = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x = Conv2D(32, kernel_size = (3,3), strides= (2,2), padding = 'same', activation='relu')(x)
  x = BatchNormalization()(x)
  
##Max Dilation rate will be vary in the range (1,4). 

# Max Dilation rate is 4 for tensor (32x32x32)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=4, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=4)


# Max Dilation rate is 3 for (16x16x64)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=3, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=3)

# Max Dilation rate is 2 for (8x8x128)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=128, max_dilation=2, number_of_units=depth)

  x = GlobalAveragePooling2D()(x)

  x = Dense(64, activation='relu')(x)
  x = Dense(nb_class, activation= 'softmax')(x)

  model = Model(xin, x)

  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  return model
  
 

## CovXNet32:

def CovXNet32(input_shape, nb_class, depth):
  
  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate
  from keras.layers import MaxPool2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop
  
  def Residual_Unit(input_tensor, nb_of_input_channels, max_dilation, number_of_units):

    for i in range(number_of_units):
      x1 = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
      x1 = BatchNormalization()(x1)

      a = []

      for i in range(1, max_dilation+1):
        temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
        temp = BatchNormalization()(temp)
        a.append(temp)

      x = Concatenate(axis= -1)(a)
      x = Conv2D(nb_of_input_channels, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
      x = BatchNormalization()(x)

      x = Add()([x, input_tensor])

      input_tensor = x

    return x
  
  def Shifter_Unit(input_tensor, nb_of_input_channels, max_dilation):

    x1 = Conv2D(nb_of_input_channels*4, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(input_tensor)
    x1 = BatchNormalization()(x1)

    a = []

    for i in range(1, max_dilation+1):
      temp = DepthwiseConv2D( kernel_size=(3,3), dilation_rate = (i,i), padding = 'same', activation= 'relu')(x1)
      temp = MaxPool2D(pool_size=(2,2))(temp)
      temp = BatchNormalization()(temp)
      a.append(temp)

    x = Concatenate(axis= -1)(a)

    x = Conv2D(nb_of_input_channels*2, kernel_size = (1,1), strides = (1,1), padding='same', dilation_rate= (1,1), activation='relu')(x)
    x = BatchNormalization()(x)

    return x
  
  
 
  xin = Input(shape= input_shape)

  x = Conv2D(16, kernel_size = (5,5), strides= (1,1), padding = 'same', activation='relu')(xin)
  x = BatchNormalization()(x)

  x = Conv2D(32, kernel_size = (3,3), strides= (2,2), padding = 'same', activation='relu')(x)
  x = BatchNormalization()(x)
  
##Max Dilation rate will be vary in the range (1,3). 

# Max Dilation rate is 3 for tensor (16x16x32)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=3, number_of_units=depth)
  x = Shifter_Unit(input_tensor=x, nb_of_input_channels=32, max_dilation=3)

# Max Dilation rate is 2 for (8x8x64)
  x = Residual_Unit(input_tensor=x, nb_of_input_channels=64, max_dilation=2, number_of_units=depth)

  x = GlobalAveragePooling2D()(x)

  x = Dense(64, activation='relu')(x)
  x = Dense(nb_class, activation= 'softmax')(x)

  model = Model(xin, x)

  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

  return model


def Unet(input_shape=(256, 256, 3)):
  
  from .segmentation import bce_dice_loss, dice_coef, jaccard
  from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate, Conv2DTranspose
  from keras.layers import MaxPooling2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, concatenate
  from keras.models import Model, load_model
  from keras.utils import plot_model
  from keras.optimizers import Adam, SGD, RMSprop  
  
  inputs = Input(shape = input_shape)

  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

  conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

  model = Model(inputs=[inputs], outputs=[conv10])
  
  model.compile(loss = bce_dice_loss, metrics = [dice_coef, jaccard], optimizer = Adam(lr = 1e-4))
  
  return model
  
# Unet++ without any extra convolution

def UnetPlus(input_shape = (256, 256, 3), summary=False):

    from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate, Conv2DTranspose
    from keras.layers import MaxPooling2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, concatenate, UpSampling2D
    from keras.models import Model, load_model
    from keras.utils import plot_model
    from keras.optimizers import Adam, SGD, RMSprop
    from .segmentation import bce_dice_loss, dice_coef, jaccard
    
    inputs = Input(shape = input_shape)

    l_0_0 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    l_0_0 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(l_0_0)

    l_1_0 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    l_1_0 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(l_1_0)

    l_0_1 = concatenate([l_0_0, UpSampling2D(size=(2,2))(l_1_0)], axis=3)#
    l_0_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_1)# extra conv

    l_2_0 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    l_2_0 = Conv2D(128, (3, 3), activation='relu', padding='same')(l_2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(l_2_0)

    l_1_1 = concatenate([l_1_0, UpSampling2D(size=(2,2))(l_2_0)], axis=3)#
    l_1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_1)# extra conv

    l_0_2 = concatenate([l_0_1, UpSampling2D(size=(2,2))(l_1_1)], axis=3)##
    l_0_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_2)## extra conv

    l_3_0 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    l_3_0 = Conv2D(256, (3, 3), activation='relu', padding='same')(l_3_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(l_3_0)

    l_2_1 = concatenate([l_2_0, UpSampling2D(size=(2,2))(l_3_0)], axis=3)#
    l_2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(l_2_1)# extra conv

    l_1_2 = concatenate([l_1_1, UpSampling2D(size=(2,2))(l_2_1)], axis=3)##
    l_1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_2)## extra conv

    l_0_3 = concatenate([l_0_2, UpSampling2D(size=(2,2))(l_1_2)], axis=3)###
    l_0_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_3)### extra conv

    l_4_0 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    l_4_0 = Conv2D(512, (3, 3), activation='relu', padding='same')(l_4_0)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(l_4_0), l_3_0], axis=3)
    l_3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    l_3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(l_3_1)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(l_3_1), l_2_1], axis=3)
    l_2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    l_2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(l_2_2)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(l_2_2), l_1_2], axis=3)
    l_1_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    l_1_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l_1_3), l_0_3], axis=3)
    l_0_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    l_0_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_4)

    l_0_4 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_4)
    l_0_3 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_3)#extra conv
    l_0_2 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_2)#extra conv
    l_0_1 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_1)#extra conv

    outputs = concatenate([l_0_1, l_0_2, l_0_3, l_0_4], axis=3)#
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outputs)

    model = Model(inputs=[inputs], outputs=outputs)

    model.compile(loss = bce_dice_loss, metrics = [dice_coef, jaccard], optimizer = Adam(lr = 1e-4))

    if summary:
      model.summary()

    return model
  
  
# Unet++
  
def UnetPlusPlus(input_shape = (256, 256, 3), summary = False):

    from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, SeparableConv2D, Concatenate, Conv2DTranspose
    from keras.layers import MaxPooling2D, Input, Add, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, concatenate, UpSampling2D
    from keras.models import Model, load_model
    from keras.utils import plot_model
    from keras.optimizers import Adam, SGD, RMSprop
    from .segmentation import bce_dice_loss, dice_coef, jaccard

    inputs = Input(shape = input_shape)

    l_0_0 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    l_0_0 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(l_0_0)

    l_1_0 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    l_1_0 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(l_1_0)

    l_0_1 = concatenate([l_0_0, UpSampling2D(size=(2,2))(l_1_0)], axis=3)#
    l_0_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_1)# extra conv

    l_2_0 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    l_2_0 = Conv2D(128, (3, 3), activation='relu', padding='same')(l_2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(l_2_0)

    l_1_1 = concatenate([l_1_0, UpSampling2D(size=(2,2))(l_2_0)], axis=3)#
    l_1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_1)# extra conv

    l_0_2 = concatenate([l_0_1, l_0_0, UpSampling2D(size=(2,2))(l_1_1)], axis=3)##
    l_0_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_2)## extra conv

    l_3_0 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    l_3_0 = Conv2D(256, (3, 3), activation='relu', padding='same')(l_3_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(l_3_0)

    l_2_1 = concatenate([l_2_0, UpSampling2D(size=(2,2))(l_3_0)], axis=3)#
    l_2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(l_2_1)# extra conv

    l_1_2 = concatenate([l_1_1, l_1_0, UpSampling2D(size=(2,2))(l_2_1)], axis=3)##
    l_1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_2)## extra conv

    l_0_3 = concatenate([l_0_2, l_0_1, l_0_0, UpSampling2D(size=(2,2))(l_1_2)], axis=3)###
    l_0_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_3)### extra conv


    l_4_0 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    l_4_0 = Conv2D(512, (3, 3), activation='relu', padding='same')(l_4_0)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(l_4_0), l_3_0], axis=3)
    l_3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    l_3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(l_3_1)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(l_3_1), l_2_1, l_2_0], axis=3)
    l_2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    l_2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(l_2_2)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(l_2_2), l_1_2, l_1_1, l_1_0], axis=3)
    l_1_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    l_1_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(l_1_3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(l_1_3), l_0_3, l_0_2, l_0_1, l_0_0], axis=3)
    l_0_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    l_0_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(l_0_4)

    l_0_4 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_4)
    l_0_3 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_3)#extra conv
    l_0_2 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_2)#extra conv
    l_0_1 = Conv2D(1, (1, 1), activation='sigmoid')(l_0_1)#extra conv

    outputs = concatenate([l_0_1, l_0_2, l_0_3, l_0_4], axis=3)#
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outputs)

    model = Model(inputs=[inputs], outputs=outputs)

    model.compile(loss = bce_dice_loss, metrics = [dice_coef, jaccard], optimizer = Adam(lr = 1e-4))

    if summary:
      model.summary()

    return model
