# Deep Generative Model for Single Image De-hazing on Embedded Platform
## Code base for our paper titled "Deep Generative Model for Single Image De-Hazing on Embedded Platform", 2020 IEEE Region 10 Symposium, Dhaka, Bangladesh.
## Requirements
1. Tensorflow-gpu v2
2. Pillow(PIL)
3. Matplotlib
## Content
This repository contains code for training, testing and evaluation. Pre trained model will be added soon.
> 1. train.py - You can train the model using this file. Set the different paratemeters like epoch, batch size, learning rate as per your preference.
> 2. test.py - Code for testing the trained model. You can enter the location of testing folder and all the images would be inferenced and saved in different directory.
> 3. utils/PSNR_SSIM.m - MATPLAB code for calculating average PSNR and SSIM values. Displays average values of PSNR and SSIM.
> 4. utils/augmentation.py - Code for data augmentation. Can generate 9 images for one image. Can be useful if dataset is small.
> 5. utils.resizeConcat.py - Code for resizing images to 256*256 and concatenating for training.
> 6. densenet56.py - Code  containing densenet implementation used in the paper.
> 7. TensorRT.ipynb - Notebook for converting keras' .h5 model to TensorRT optimized graph
> 8. JetsonNano.ipynb - Notebook for testing model on Jetson Nano.
## Acknowledgement
This code is based on [pix2pix Tensorflow](https://www.tensorflow.org/tutorials/generative/pix2pix) for training code and [DLology](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/) for TensorRT and for Inference part.
