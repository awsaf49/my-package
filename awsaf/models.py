# Network 2

def Network2(Input_shape):

  from keras.layers import *
  from keras.models import *
  from keras.utils import *
  
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
  
  from keras.layers import *
  from keras.models import *
  from keras.utils import *
  
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
  
  from keras.layers import *
  from keras.models import *
  from keras.utils import *
  
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
  
  from keras.layers import *
  from keras.models import *
  from keras.utils import *
  
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
  
  from keras.layers import *
  from keras.models import *
  from keras.utils import *
  
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
