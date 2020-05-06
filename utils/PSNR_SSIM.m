clc
close all
close all
cd('C:\Users\Raghu\Downloads\PSNR_FSIM\');
% dir testingImages
list1 = dir('groundtruth_ohaze');
list1 = {list1.name};
ssim_ans = [];
psnr_ans = [];
%%Enter the directories of groundtruth and predicted images.
for i= 3: 1: 10
 groundtruthlocation = strcat('TesingImage_location', list1(i));
    predictedlocation = strcat('PredictedImage Location', list1(i));
    
    testImage = imread(groundtruthlocation{:});
    predictedImage = imread(predictedlocation{:});
    psnr_ans(i-2) = psnr(predictedImage,testImage);
    ssim_ans(i-2) = ssim(predictedImage,testImage);
end

display(mean(psnr_ans))
display(mean(ssim_ans))
