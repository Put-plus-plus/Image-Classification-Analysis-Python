## Description 
Binary image classification is the assignment of images into one of two categories based on specific rules. In this case the classification analysis is organised into the `BinaryClassifier` class, which has five methods. The `prepare_images()` method accepts a top-folder that contains two uniquely named sub-folders with negative and positive images (eg 'normal' and tumour'), and outputs a single folder with consistently labelled and formatted negative and positive images (eg 'normal27', 'tumour49'). The `prepare_arrays()` method accepts the folder with consistently labelled and formatted positive and negative images, and outputs four numpy arrays with training and testing data. The `define_model()` method defines the structure of the convolutional neural network model. The `fit_model()` method trains the convolutional neural network model. The a `make_prediction()` method accepts a single image previously unseen by the convolutional neural network model, and outputs the  image labelled with its class as predicted by the model.

The image dataset used in the example below is an adapted version of Msoud Nickparvar. (2021). <i>Brain Tumor MRI Dataset</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886, and is used under a CC0: Public Domain licence.

## Dependencies  
* Microsoft Windows 10.0.19045
* Python 3.9.1
* glob (built-in), cv2 3.4.13, shutil (built-in), numpy 1.22.2, sklearn 1.0.1, keras 2.12.0, tflearn 0.5.0, matplotlib 3.3.4   

## Execution - brain MRI image example
```python
from classificationanalysis import *

mri_tot_no_images = 196
mri_image_size = 100
mri_no_channels = 1
mri_class1 = 'normal'
mri_class2 = 'tumour'

brain_mri = BinaryClassifier(mri_tot_no_images, mri_image_size, mri_no_channels, mri_class1, mri_class2)
mri_image_top_folder= 'C:\\Users\\plain\\Desktop\\brain_mri_images\\*'
brain_mri.prepare_images(mri_image_top_folder)

mri_image_folder = 'C:\\Users\\plain\\Desktop\\brain_mri_images\\*.jpg'
brain_mri.prepare_arrays(mri_image_folder)

brain_mri.define_model()

brain_mri.fit_model()

unseen_mri_image = 'C:\\Users\\plain\\Desktop\\tumour.jpg'
brain_mri.make_prediction(unseen_mri_image)
```

