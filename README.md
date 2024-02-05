## Description 
The project is a binary image classifier. The `BinaryClassifier` class. The `prepare_images()` method input is the path of a top-folder that should contain two sub-folders with unique names (eg 'normal' and tumour') and should respectively contain negative and positive jpg imagnes. The output is the top-folder, which now contains no sub-folders, only consistently labelled and formated positive and negative jpg images. The `prepare_arrays()` method input is the path of the folder with consistently labelled and formated positive and negative jgp images, and the output is four numpy arrays, two of which contain training data and two of which contain testing data (STATE WHAT'S IN THE ARRAYS - LABELLES AND IMAGES). The `define_model()` method defines the structure of a convolutional neural network model. The `fit_model()` method trains the convolutional neural network model using the structure defined by `define_model()`and the data prepared by `prepare_arrays()`. The a `make_prediction()` method Input is the path of a single jpg image previously unseen by the convolutional neural network model, and output is the jpg image labeled with its predicted image class as predicted by the cnn model.

Image dataset used in the below example is a selection of the images in the Msoud Nickparvar. (2021). <i>Brain Tumor MRI Dataset</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2645886. Used and shared under CC0: Public Domain licence.

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

## Animation - brain MRI image example
remember to add the link to the GIF, which I must also make sure to add to the repo
