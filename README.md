## Description 
The project is a binary image classifier. The `BinaryClassifier` class. The `prepare_images()` method input is the path of a top-folder that should contain two sub-folders with unique names (eg 'normal' and tumour') and should respectively contain negative and positive jpg imagnes. The output is the top-folder, which now contains no sub-folders, only consistently labelled and formated positive and negative jpg images. The `prepare_arrays()` method input is the path of the folder with consistently labelled and formated positive and negative jgp images, and the output is four numpy arrays, two of which contain training data and two of which contain testing data (STATE WHAT'S IN THE ARRAYS - LABELLES AND IMAGES). The `define_model()` method defines the structure of a convolutional neural network model. The `train_model()` method trains the convolutional neural network model using the structure defined by `define_model()`and the data prepared by `prepare_arrays()`. The a `make_prediction()` method Input is the path of a single jpg image previously unseen by the convolutional neural network model, and output is the jpg image labeled with its predicted image class as predicted by the cnn model.

# IMAGE DATA

## Dependencies  
* Microsoft Windows 10.0.19045
* Python 3.9.1
* numpy 1.22.2, cv2 3.4.13, sklearn 1.0.1, glob KEEP GOING 

## Execution - brain tumour MRI example BETTER NAMING IS NEEDED
```python
new_no_images = 196
new_image_size = 100
new_no_channels = 1
new_image_class1 = 'normal'
new_image_class2 = 'tumour'
new_model = BinaryImageClassifier(new_no_images, new_image_size, new_no_channels, new_image_class1, new_image_class2)

new_top_folder_path = 'C:\\Users\\plain\\Desktop\\brain_mri_images\\*'
new_model.image_preparation(new_top_folder_path)
new_folder_path = 'C:\\Users\\plain\\Desktop\\brain_mri_images\\*.jpg'
new_model.array_preparation(new_folder_path)

new_model.convul_model_structure()
new_model.convul_model_performance()
new_unseen_image_path = 'C:\\Users\\plain\\Desktop\\test_tumour.jpg'
new_model.convul_model_prediction(new_unseen_image_path)
```

## Animation - brain tumour MRI example 
remember to add the link to the GIF, which I must also make sure to add to the repo
