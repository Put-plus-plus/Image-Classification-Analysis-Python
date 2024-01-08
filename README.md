## Description 
The project involves processing images so that the format is conducive for downstream analysis. Then go on and build a neural network and train the model, and then test out new images to see if the trained model works. 

The `ImageClassifier` class has a `image_preparation()` method for reseizing and recolouring images, a `array_preparation()` method from extracting x from the images  preparing test and train data sets, a `convul_model_structure()` method for defining the structure of the convul netwoek, a `convul_model_performance()` method for training the model, and a `convul_model_prediction()` method for testing the prediction of the model.  


## Dependencies
* Microsoft Windows version 10.0.19045
* Python version 3.9.1
* Numpy, cv2, skleran, glob, random, tflearn, keras, matplotlib, shutil # cv2 or that other name?

## Execution - Tumour Head MRI Example 
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

## Animation - Tumour Head MRI Example
remember to add the link to the GIF, which I must also make sure to add to the repo, see stackoverflow - maybe show the test image 
