# Luna2016-Lung-Nodule-Detection
Course Project for Bio Medical Imaging: Detecting Lung Nodules from CT Scans

This project was inspired by the LUNA 2016 Grand Challenge that uses low-dose CT scans for pulmonary nodule detection in lung scans.

We use a U-net Convolutional Neural Network for lung nodule detection.

# Running The Code
==================================
## Dependencies
### Preprocess
* SimpleITK
* numpy
* csv
* glob
* pandas
* scikit-learn
* scikit-image

### UNET
* theano
* keras
---
## Preprocessing
---
TODO : UPDATE

## Network
----
LUNA_unet.py contains the code for the actual model. The following arguments can be passed to the network.

    -batch_size     (default 2,batch size)
    -lr      		(default 0.001, learning rate)
    -load_weights	(default False, load pre-existing model)
    -filter_width	(default 3, The default filter width)
    -stride			(default 3, The stride of the filters)
    -model_file		(default '', The path to the model file )
    -save_prefix	(default model prefix, The prefix of the saving of models)
 The model expects as inputs the segmented lung images (512x512) as inputs (generated in preprocessing steps) and generates the mask of the nodule (512x512). The loss used is the dice coefficient loss between the predicted mask and the gold mask. 
 Model weights are saved using the following convention:
 model\_\_script_on_epoch\_ + _epoch number_  + \_lr\_ + _lr_ + \_WITH_STRIDES\_ + _filter stride_ + \_FILTER\_WIDTH\_ + _filter width_ + .weights

 # Results 
 =========
 TODO : UPDATE
