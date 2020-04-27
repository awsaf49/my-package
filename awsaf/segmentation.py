# train generator for image segmentation from dataframe

from keras.preprocessing.image import ImageDataGenerator

def image_generator_df(df,
                       image_path ,
                       mask_path,
                       target_size ,
                       batch_size, 
                       args = dict(),
                       image_normalize = True,
                       mask_normalize = True):
                       
                       
     # args = are the parameters of image datagenerator for augmentation
    
     """ Example of args:
     args = dict(  zoom_range  = 0.2,
                shear_range     = 0,
                rotation_range  = 20,
                width_shift_range  = 0.2,
                height_shift_range = 0.2,
                fill_mode = 'constant')
      """
                 
    
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







