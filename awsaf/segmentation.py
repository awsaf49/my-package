# train generator for image segmentation from dataframe
import keras.backend as K
from keras.losses import binary_crossentropy


# Losses for segmentation

def dice_coef(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def jaccard(y_true, y_pred):
    
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    
    return intersection / union


# args = are the parameters of image datagenerator for augmentation

#  """ 
#  Example of args:
#  args = dict(  zoom_range  = 0.2,
#             shear_range     = 0,
#             rotation_range  = 20,
#             width_shift_range  = 0.2,
#             height_shift_range = 0.2,
#             fill_mode = 'constant')
            
#             """

def image_generator_df(df,
                       image_path ,
                       mask_path,
                       target_size ,
                       batch_size, 
                       args = dict(),
                       image_normalize = True,
                       mask_normalize = True):
                 
    
    from keras.preprocessing.image import ImageDataGenerator
    seed = 101
    image_gen =  ImageDataGenerator(**args )

    mask_gen  = ImageDataGenerator(**args )

    def adjust_data(image, mask, image_normalize, mask_normalize):
        
        if (image_normalize==True):
            image = image / 255.
        if (mask_normalize == True):
            mask = mask /255.
            
            
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return (image, mask)

    image_generator = image_gen.flow_from_dataframe(df, 
                                                    directory=None,
                                                    x_col= image_path,
                                                    target_size= target_size,
                                                    color_mode='rgb', 
                                                    class_mode= None,
                                                    batch_size= batch_size, 
                                                    shuffle=False,
                                                    seed= seed
                                                   )

    mask_generator = mask_gen.flow_from_dataframe(  df, 
                                                    directory= None,
                                                    x_col= mask_path,
                                                    target_size= target_size,
                                                    color_mode='grayscale', 
                                                    class_mode= None,
                                                    batch_size= batch_size, 
                                                    shuffle=False,
                                                    seed= seed
                                                   )

    gen = zip(image_generator, mask_generator)
    
    for (image, mask) in gen:
      image, mask = adjust_data(image, mask, image_normalize, mask_normalize)
      yield (image, mask)





# image data generator from directory

# args = dict(  horizontal_flip = True,
#               zoom_range      = 0.1,
#               shear_range     = 0,
#               rotation_range  = 10,
#               width_shift_range  = 0.1,
#               height_shift_range = 0.1,
#               fill_mode = 'constant')

def image_generator_dir(
                       image_dir,
                       mask_dir,
                       target_size ,
                       batch_size, 
                       args = dict(),
                       image_normalize = True,
                       mask_normalize = True):
    
    
    from keras.preprocessing.image import ImageDataGenerator
    seed = 101
    image_gen =  ImageDataGenerator(**args )

    mask_gen  = ImageDataGenerator(**args )

    def adjust_data(image, mask, image_normalize, mask_normalize):
        
        if (image_normalize==True):
            image = image / 255.
        if (mask_normalize == True):
            mask = mask /255.
            
            
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return (image, mask)

    image_generator = image_gen.flow_from_directory(directory = image_dir,
                                                    class_mode =None,
                                                    shuffle = False,
                                                    batch_size = batch_size,
                                                    target_size = target_size,
                                                    seed=seed
                                                   )

    mask_generator = mask_gen.flow_from_directory(  directory = mask_dir,
                                                    class_mode =None,
                                                    shuffle = False,
                                                    batch_size = batch_size,
                                                    target_size = target_size,
                                                    seed=seed
                                                   )

    gen = zip(image_generator, mask_generator)
    
    for (image, mask) in gen:
      image, mask = adjust_data(image, mask, image_normalize, mask_normalize)
      yield (image, mask)




# epoch plot


def create_epoch_plot_model(model):
            
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    dice = model.history.history['dice_coef']
    val_dice = model.history.history['val_dice_coef']
    jacc = model.history.history['jaccard']
    val_jacc = model.history.history['val_jaccard']

    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    epochs = range(len(dice))
    fig = plt.gcf()
    fig.set_size_inches(16, 8)

    plt.plot(epochs, dice, label='Training Dice',marker = "o")
    plt.plot(epochs, val_dice, label='Validation Dice',marker = "o")

    plt.plot(epochs, jacc, label='Training Jaccard',marker = "o")
    plt.plot(epochs, val_jacc, label='Validation Jaccard',marker = "o")

    plt.title('Training and validation Accurcay')
    plt.xticks(np.arange(0, len(dice), 10))
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.figure()

    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    plt.plot(epochs, loss,  label='Training Loss',marker = "o")
    plt.plot(epochs, val_loss, label='Validation Loss',marker = "o")
    plt.title('Training and validation Loss')
    plt.xticks(np.arange(0, len(dice), 10))
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.figure()
    plt.show()

    
# epoch plot from dataframe

def create_epoch_plot_df(df):
            
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    dice = df['dice_coef']
    val_dice = df['val_dice_coef']
    jacc = df['jaccard']
    val_jacc = df['val_jaccard']

    loss = df['loss']
    val_loss = df['val_loss']

    epochs = range(len(dice))
    fig = plt.gcf()
    fig.set_size_inches(16, 8)

    plt.plot(epochs, dice, label='Training Dice',marker = "o")
    plt.plot(epochs, val_dice, label='Validation Dice',marker = "o")

    plt.plot(epochs, jacc, label='Training Jaccard',marker = "o")
    plt.plot(epochs, val_jacc, label='Validation Jaccard',marker = "o")

    plt.title('Training and validation Accurcay')
    plt.xticks(np.arange(0, len(dice), 10))
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.figure()

    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    plt.plot(epochs, loss,  label='Training Loss',marker = "o")
    plt.plot(epochs, val_loss, label='Validation Loss',marker = "o")
    plt.title('Training and validation Loss')
    plt.xticks(np.arange(0, len(dice), 10))
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.figure()
    plt.show()
    
    
    
    

def visualize(model, test_generator, n_outputs = 5):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style = 'white')

    test_image, test_mask = test_generator.__next__()

    test_mask = np.reshape(test_mask, (test_mask.shape[0], test_mask.shape[1], test_mask.shape[2]))

    pred_mask = model.predict(test_image, verbose = 1)
    pred_mask = np.reshape(pred_mask, (pred_mask.shape[0], pred_mask.shape[1], pred_mask.shape[2]))

    pred_mask[pred_mask>0.5] =  1
    pred_mask[pred_mask<=0.5] = 0

    fig, ax = plt.subplots(n_outputs, 3, figsize = (15, 3*n_outputs),constrained_layout=True)

    for idx in tqdm(range(n_outputs)):

        ax[idx][0].imshow(test_image[idx,:,:,0], cmap = 'bone')
        ax[idx][0].set_title('Ground_truth:')

        ax[idx][1].imshow(test_image[idx,:,:,0], cmap = 'bone')
        mask = test_mask[idx,:,:]
        mask = np.ma.masked_where(mask==0, mask)
        ax[idx][1].imshow(mask, cmap = 'autumn', alpha = 0.7)
        ax[idx][1].set_title('Segmentaton:')

        ax[idx][2].imshow(test_image[idx,:,:,0], cmap = 'bone')
        p_mask = pred_mask[idx,:,:]
        p_mask = np.ma.masked_where(p_mask==0, p_mask)
        ax[idx][2].imshow(p_mask, cmap = 'winter', alpha = 0.8)
        ax[idx][2].set_title('Prediction:')
    

