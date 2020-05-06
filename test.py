# -*- coding: utf-8 -*-
from __future__ import print_function, division
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import os

#Enter directory of saved model here

dehaze_model = load_model('models/OHAZE_epoch119.h5')
#Enter directory of testing images. Download O-HAZE dataset, resize the images to 256,256 and place it in testing folder
list_dir = os.listdir('testingDirectory')
inference_time = []
for i in list_dir:
    img = image.load_img('testingDirectory/{}'.format(i), target_size = (256,256))
    img1 = image.img_to_array(img)
    img1 = np.expand_dims(img1, axis=0)
    img1 = img1/127.5 - 1.
    stime = time.time()
    dehazed_image = dehaze_model.predict(img1)
    total_time = time.time() - stime
    inference_time.append(total_time)
    dehazed_image = 0.5*dehazed_image + 0.5
    plt.imshow(dehazed_image[0])
    dehazed_image = np.reshape(dehazed_image, (256,256,3))
    img = Image.fromarray((dehazed_image*255).astype(np.uint8))
    #Enter the directory where you wish to save the results
    img.save('results/{}'.format(i))
print("Time taken for inference {} seconds".format(np.array(inference_time).mean()))
