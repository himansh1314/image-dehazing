
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import numpy as np

list_dir_GT = os.listdir('GT')
list_dir_HAZY = os.listdir('hazy')

def load_image(path):
    img = image.load_img(path)
    img1 = image.img_to_array(img)
    return img1
        
for i in range(0,46):
    image_hazy = load_image('hazy/{}'.format(list_dir_HAZY[i]))
    image_GT = load_image('GT/{}'.format(list_dir_GT[i]))
    
    original_hazy_crop = tf.image.central_crop(image_hazy, central_fraction = 0.5).numpy()
    original_gt_crop = tf.image.central_crop(image_GT, central_fraction = 0.5).numpy()
    
    original_hazy_flip = tf.image.flip_left_right(image_hazy).numpy()
    original_gt_flip = tf.image.flip_left_right(image_GT).numpy()
    
    crop_hazy_flip = tf.image.flip_left_right(original_hazy_crop).numpy()
    crop_gt_flip = tf.image.flip_left_right(original_gt_crop).numpy()
    
    original_hazy_rotation = tf.keras.preprocessing.image.apply_affine_transform(image_hazy,theta=180 )
    original_gt_rotation  = tf.keras.preprocessing.image.apply_affine_transform(image_GT,theta=180)
    
    original_hazy_shift= tf.keras.preprocessing.image.apply_affine_transform(image_hazy,tx=30,ty=30 )
    original_gt_shift  = tf.keras.preprocessing.image.apply_affine_transform(image_GT,tx=30,ty=30)
    
    original_hazy_scaling= tf.keras.preprocessing.image.apply_affine_transform(image_hazy,zx=0.3,zy= 0.3 )
    original_gt_scaling  = tf.keras.preprocessing.image.apply_affine_transform(image_GT,zx=0.3,zy=0.3)
    
    scaled_hazy_rot= tf.keras.preprocessing.image.apply_affine_transform(original_hazy_scaling,theta=180)
    scaled_gt_rot  = tf.keras.preprocessing.image.apply_affine_transform(original_gt_scaling,theta=180)
    
    crop_hazy_flip_scaling =  tf.keras.preprocessing.image.apply_affine_transform(crop_hazy_flip,zx=0.3,zy= 0.3 )
    crop_gt_scaling =  tf.keras.preprocessing.image.apply_affine_transform(crop_gt_flip,zx=0.3,zy= 0.3 )
    #Saving hazy images 
    img1_hazy = Image.fromarray(image_hazy.astype(np.uint8))
    img1_hazy.save('augmented_dataset/hazy/outdoor_{}_1.png'.format(i))
    
    img2_hazy = Image.fromarray(original_hazy_crop.astype(np.uint8))
    img2_hazy.save('augmented_dataset/hazy/outdoor_{}_2.png'.format(i))
    
    img3_hazy = Image.fromarray(original_hazy_flip.astype(np.uint8))
    img3_hazy.save('augmented_dataset/hazy/outdoor_{}_3.png'.format(i))
    
    img4_hazy = Image.fromarray(crop_hazy_flip.astype(np.uint8))
    img4_hazy.save('augmented_dataset/hazy/outdoor_{}_4.png'.format(i))
    
    img5_hazy = Image.fromarray(original_hazy_rotation.astype(np.uint8))
    img5_hazy.save('augmented_dataset/hazy/outdoor_{}_5.png'.format(i))
    
     
    img6_hazy = Image.fromarray(original_hazy_shift.astype(np.uint8))
    img6_hazy.save('augmented_dataset/hazy/outdoor_{}_6.png'.format(i))
    
    img7_hazy = Image.fromarray(original_hazy_scaling.astype(np.uint8))
    img7_hazy.save('augmented_dataset/hazy/outdoor_{}_7.png'.format(i))
    
    img8_hazy = Image.fromarray(scaled_hazy_rot.astype(np.uint8))
    img8_hazy.save('augmented_dataset/hazy/outdoor_{}_8.png'.format(i))
    
    img9_hazy = Image.fromarray(crop_hazy_flip_scaling.astype(np.uint8))
    img9_hazy.save('augmented_dataset/hazy/outdoor_{}_9.png'.format(i))
    
    #Saving ground truth image
    
    img1_gt = Image.fromarray(image_GT.astype(np.uint8))
    img1_gt.save('augmented_dataset/GT/outdoor_{}_1.png'.format(i))
    
    img2_gt = Image.fromarray(original_gt_crop.astype(np.uint8))
    img2_gt.save('augmented_dataset/GT/outdoor_{}_2.png'.format(i))
    
    img3_gt = Image.fromarray(original_gt_flip.astype(np.uint8))
    img3_gt.save('augmented_dataset/GT/outdoor_{}_3.png'.format(i))
    
    img4_gt = Image.fromarray(crop_gt_flip.astype(np.uint8))
    img4_gt.save('augmented_dataset/GT/outdoor_{}_4.png'.format(i))
    
    img5_gt = Image.fromarray( original_gt_rotation.astype(np.uint8))
    img5_gt.save('augmented_dataset/GT/outdoor_{}_5.png'.format(i))
                      

    
    img6_gt = Image.fromarray( original_gt_shift.astype(np.uint8))
    img6_gt.save('augmented_dataset/GT/outdoor_{}_6.png'.format(i))
    
    img7_gt = Image.fromarray( original_gt_scaling.astype(np.uint8))
    img7_gt.save('augmented_dataset/GT/outdoor_{}_7.png'.format(i))
    
    img8_gt = Image.fromarray( scaled_gt_rot.astype(np.uint8))
    img8_gt.save('augmented_dataset/GT/outdoor_{}_8.png'.format(i))
    
    img9_gt = Image.fromarray( crop_gt_scaling .astype(np.uint8))
    img9_gt.save('augmented_dataset/GT/outdoor_{}_9.png'.format(i))
    #original_hazy_side = tf.image.crop_to_bounding_box(image_hazy, 0,0,0.5,0.5)
    
    
