# plot confusion matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas.util.testing as tm
from sklearn import metrics
import seaborn as sns
sns.set()

plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save = False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(b=False)
    if save == True:
      plt.savefig('Confusion Matrix.png', dpi = 900)
    


    
    
    

# test model performance
from datetime import datetime
import matplotlib.pyplot as plt


def test_model(model, test_generator, y_test, class_labels, cm_normalize=True, \
                 print_cm=True):
    
    # BS = 16
    results = dict()
    
    # n = len(testy)// BS

    # testX = testX[:BS*n]
    # testy = testy[:BS*n]

    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred_original = model.predict_generator(test_generator,verbose=1)
    # y_pred = (y_pred_original>0.5).astype('int')

    y_pred = np.argmax(y_pred_original, axis = 1)
    # y_test = np.argmax(testy, axis= 1)
    #y_test = np.argmax(testy, axis=-1)
    
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
    y_test = y_test.astype(int) # sparse form not categorical
    

    # balanced_accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print('---------------------')
    print('| Balanced Accuracy  |')
    print('---------------------')
    print('\n    {}\n\n'.format(balanced_accuracy))

    
    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    

    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    
    
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(8,6))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix')
    plt.show()
    

    
    # add the trained  model to the results
    results['model'] = model
    
    return
  
 




# MyLogger
      
from keras.callbacks import Callback
class MyLogger(Callback):
  
  def __init__(self, test_generator, y_test, class_labels):
    super(MyLogger, self).__init__()
    self.test_generator = test_generator
    self.y_test = y_test
    self.class_labels = class_labels
        
  def on_epoch_end(self, epoch, logs=None):
    test_model(self.model, self.test_generator, self.y_test, self.class_labels)

    
    
    

    
    
    
#  create epoch plot from dataframe

def create_epoch_plot_df(df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='darkgrid')
    
    fig, ax = plt.subplots(1, 2, figsize = (24, 8))
    
    
    acc = df['acc']
    val_acc = df['val_acc']
    loss = df['loss']
    val_loss = df['val_loss']

    epochs = range(len(acc))

    ax[0].plot(epochs, acc, 'r', label='Training accuracy',marker = "o")
    ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy',marker = "o")
    ax[0].set_title('Training and validation accuracy')
    ax[0].set_xticks(np.arange(0, len(acc), 10))
    ax[0].set_xlabel('Epoch')
    ax[0].legend(loc=0)

    ax[1].plot(epochs, loss, 'r', label='Training Loss',marker = "o")
    ax[1].plot(epochs, val_loss, 'b', label='Validation Loss',marker = "o")
    ax[1].set_title('Training and validation Loss')
    ax[1].set_xticks(np.arange(0, len(acc), 10))
    ax[1].legend(loc=0)
    ax[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()
    
    
    
    
#  create epoch plot from model

def create_epoch_plot_model(model):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='darkgrid')
    
    fig, ax = plt.subplots(1, 2, figsize = (24, 8))
    
    
    acc = model.history.history['acc']
    val_acc = model.history.history['val_acc']
    loss = model.history.history['loss']
    val_loss = model.histroy.history['val_loss']

    epochs = range(len(acc))

    ax[0].plot(epochs, acc, 'r', label='Training accuracy',marker = "o")
    ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy',marker = "o")
    ax[0].set_title('Training and validation accuracy')
    ax[0].set_xticks(np.arange(0, len(acc), 10))
    ax[0].set_xlabel('Epoch')
    ax[0].legend(loc=0)

    ax[1].plot(epochs, loss, 'r', label='Training Loss',marker = "o")
    ax[1].plot(epochs, val_loss, 'b', label='Validation Loss',marker = "o")
    ax[1].set_title('Training and validation Loss')
    ax[1].set_xticks(np.arange(0, len(acc), 10))
    ax[1].legend(loc=0)
    ax[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()    
    
        
  
  
  
  
# layerwise finetunning

from keras.optimizers import Adam, RMSprop
from tqdm import tqdm
from keras.utils.layer_utils import count_params

# model, epochs, ordered_layre_name, lr, reduce
# epochs = 50
# lr = 1e-2
# reduce =1
# ordered_layers_name = ["block5_conv3","block4_conv3","block3_conv3","block2_conv2","block1_conv2"]
def layer_wise_training(model= [],
                        loss = 'categorical_crossentropy',
                        metrics =['acc'],
                        epochs =[1],
                        lr =[1e-1],
                        callbacks = [],
                        class_weight = [],
                        train_generator = [],
                        test_generator =[],
                        ordered_layers_name =[] # from top(Dense) to bottom(Input)
                       ):
  
  if len(epochs) != len(ordered_layers_name):
    print('Input error!!!')
    
  if (train_generator == []) or (test_generator == []) or (model==[]) :
    print('Generator error!!!')
      
      
   

  print('\nLayerWise Training\n')



#     print(lr)

  initial_epoch = 0
  for idx, layer_name in enumerate(tqdm(ordered_layers_name)):


      # Learning Rate stays same for first epoch
#         if idx == 0 :
#             lr = lr
#         else:
#             lr = lr*reduce


      # UnFreezeing All the layers
      for x in range(len(model.layers)):
        model.layers[x].trainable = True


      # Finding layer index for Unfreeze
      layer_name = layer_name
      index = None
      for x in range(len(model.layers)):
          if model.layers[x].name == layer_name:
              index = x

      # Freezeing the layers
      fine_tune_at = index
      for x in range(fine_tune_at+1):
        model.layers[x].trainable = False
  #       print(model.layers[x], model.layers[x].trainable)


      # Compiling the model
       
      
      model.compile(optimizer= Adam(lr = lr[idx]), loss= loss , metrics= metrics)
#         print(lr*reduce)

      # Training the Model
      if class_weight == []:
            class_weight = np.full(len(np.unique(train_generator.labels)), 1,  dtype = int)
            

            
      print(f'Training Stage: {idx+1}  ||  Total Trainable Parameters: {count_params(model.trainable_weights):,d}  ||  Learning_rate: {lr[idx]:,.7f}')
      print('==================================================================================')
      print('\n')
      model.fit_generator(train_generator,
                          epochs    = initial_epoch + epochs[idx],
                          steps_per_epoch  = train_generator.samples // train_generator.batch_size,
                          validation_data  = test_generator,
                          validation_steps = test_generator.samples // test_generator.batch_size,
                          class_weight = class_weight,
                          callbacks = callbacks,
                          initial_epoch = initial_epoch)

      initial_epoch = initial_epoch + epochs[idx]

  #     print('training')
      print('============================================================================================')
      print('\n')

      
      
# finding best model from filenames with filepath (glob)

def best_model(names=[]):
  import numpy as np
  from keras.models import load_model
  
  acc = []
  for name in names:
      acc.append(float(name.split('-')[-1][:-3]))
  idx = np.argmax(acc)
  max_model_name = names[idx]


  max_model = load_model(max_model_name)
  print(f'val acc: {np.max(acc):,.3f}\n')
  return max_model


# create saliency map from image_path and model



def saliency_map(img_path, model, image_size = (256, 256)):

  # Creating Saliency Map
  from keras import backend as K
  import tensorflow as tf
  from keras.preprocessing.image import load_img, img_to_array
  from keras.models import Model, load_model
  import cv2
  import matplotlib.pyplot as plt

  img = load_img(img_path, target_size=(int(model.input.shape[1]), int(model.input.shape[2])))
  img = img_to_array(img)
  img = img/255.0
  pred_img = np.expand_dims(img, axis=0)
  pred = model.predict(pred_img)

  class_outp = model.output[:, np.argmax(pred)]
  sal = K.gradients(tf.reduce_sum(class_outp), model.input)[0]

  # Keras function returning the saliency map given an image input
  sal_fn = K.function([model.input], [sal])
  # Generating the saliency map and normalizing it
  img_sal = sal_fn([np.resize(pred_img, (1, int(model.input.shape[1]), int(model.input.shape[2]), 3))])[0]
  img_sal = np.abs(img_sal)
  img_sal /= img_sal.max()
  img_sal = cv2.resize(img_sal[0,:,:,0]*10000, dsize = image_size)
  
  return img, img_sal




# grad-cam from image_path and model and last_conv_layer_name

def grad_cam(img_path , model , last_conv_layer_name,  image_size = (256, 256), alpha = 0.4):


  import os
  import numpy as np
  import pandas as pd
  import cv2
  #   from google.colab.patches import cv2_imshow
  from PIL import Image
  from matplotlib import pyplot as plt

  from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
  from keras.preprocessing.image import load_img, img_to_array
  from keras.models import Model, load_model
  from keras import backend as K



  # ==================================
  #   1. Test images visualization
  # ==================================
  img = load_img(img_path, target_size= (int(model.input.shape[1]), int(model.input.shape[2])))
  # msk = load_img(mask_path, target_size=image_size, color_mode= 'grayscale')
  img = img_to_array(img)
  img = img/255.0
  pred_img = np.expand_dims(img, axis=0)
  # pred_img = preprocess_input(img)
  pred = model.predict(pred_img)

  # ==============================
  #   2. Heatmap visualization 
  # ==============================
  # Item of prediction vector
  pred_output = model.output[:, np.argmax(pred)]

  # Feature map of 'conv_7b_ac' layer, which is the last convolution layer
  last_conv_layer = model.get_layer(last_conv_layer_name)

  # Gradient of class for feature map output of 'conv_7b_ac'
  grads = K.gradients(pred_output, last_conv_layer.output)[0]

  # Feature map vector with gradient average value per channel
  pooled_grads = K.mean(grads, axis=(0, 1, 2))

  # Given a test image, get the feature map output of the previously defined 'pooled_grads' and 'conv_7b_ac'
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  # Put a test image and get two numpy arrays
  pooled_grads_value, conv_layer_output_value = iterate([pred_img])

  # print(pooled_grads.shape[0])
  # Multiply the importance of a channel for a class by the channels in a feature map array
  for i in range(int(pooled_grads.shape[0])):
      conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  # The averaged value along the channel axis in the created feature map is the heatmap of the class activation
  heatmap = np.mean(conv_layer_output_value, axis=-1)

  # Normalize the heatmap between 0 and 1 for visualization
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap_img = heatmap


  # =======================
  #   3. Apply Grad-CAM
  # =======================
  ori_img = load_img(img_path, target_size=image_size)

  heatmap = cv2.resize(heatmap, image_size)
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

  superimposed_img = heatmap * alpha + ori_img
  cv2.imwrite('grad_cam_result.png', superimposed_img) # otherwise it'll show error because of float
  grad_img = cv2.imread('grad_cam_result.png')
  grad_img = cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


  return ori_img, heatmap_img, grad_img
