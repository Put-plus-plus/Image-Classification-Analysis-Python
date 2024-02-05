import glob
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn import DNN
import matplotlib.pyplot as plt




class BinaryClassifier():



    def __init__(self, tot_no_images, image_size, no_channels, image_class1, image_class2):

        self.tot_no_images = tot_no_images 
        self.image_size = image_size
        self.no_channels = no_channels
        self.image_class1 = image_class1
        self.image_class2 = image_class2 

        


    def prepare_images(self, top_folder_path):                         

        '''Input is the path of a top-folder, which should contain two sub-folders. The sub-folders should have unique names 
           (eg 'normal' and 'tumour'), and contain negative and positive jpg images respectively. Output is the top-folder, which 
           now contains no sub-folders, only consistently labelled and formated positive and negative jpg images (eg normal27, tumour49).'''

        try:
            sub_folder_paths = glob.glob(top_folder_path)    
            for sub_folder_path in sub_folder_paths: 
                image_no = 1
                image_paths = glob.glob(sub_folder_path + '\\*.jpg')
                for image_path in image_paths:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
                    image = cv2.resize(image, (self.image_size, self.image_size)) 
                    cv2.imwrite(str(sub_folder_path) + str(image_no) + '.jpg', image)
                    image_no += 1
                shutil.rmtree(sub_folder_path)
            self.tot_no_images = len(glob.glob('C:\\Users\\plain\\Desktop\\brain_mri_images\\*'))
        except:
            print(f'The {BinaryClassifier.prepare_images.__name__} method could not format and label the images.')
            

   
            
    def prepare_arrays(self, folder_path): 

        '''Input is the path of a folder, which contains consistently named and formated positive and negative jgp images.
           Ouput is four appropriately formated numpy arrays, two of which contain training data (self.X_train, self.y_train) and 
           two of which contain testing data (self.X_test, self.y_test).'''
        
        try:
            image_paths = glob.glob(folder_path)  

            X_image_array = []                                         
            for image_path in image_paths:                          
                X_image_array.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)) 
            X_image_array = np.array(X_image_array).reshape(self.tot_no_images, self.image_size, self.image_size, self.no_channels)                                  

            y_label_array=[]
            for image_path in image_paths:
                if self.image_class1 in image_path:   
                    y_label_array.append(1)
                else:
                    y_label_array.append(0)        
            y_label_array = np_utils.to_categorical(y_label_array)      

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(X_image_array), np.array(y_label_array), test_size = 0.2, random_state = 42)

        except:
            print(f'The {BinaryClassifier.prepare_arrays.__name__} method could not prepare arrays containing training and testing data.')




    def define_model(self):

        '''Defines the structure of a convolutional neural network model and outputs the structure.'''

        try:

            model_learning_rate = 1e-3

            self.cnn_model_structure = input_data(shape = [self.image_size, self.image_size, self.no_channels], name = 'input')  

            self.cnn_model_structure = conv_2d(self.cnn_model_structure, 32, 5, activation = 'relu')
            self.cnn_model_structure = max_pool_2d(self.cnn_model_structure, 5)

            self.cnn_model_structure = conv_2d(self.cnn_model_structure, 64, 5, activation = 'relu')
            self.cnn_model_structure = max_pool_2d(self.cnn_model_structure, 5)

            self.cnn_model_structure = conv_2d(self.cnn_model_structure, 128, 5, activation = 'relu')
            self.cnn_model_structure = max_pool_2d(self.cnn_model_structure, 5)

            self.cnn_model_structure = conv_2d(self.cnn_model_structure, 64, 5, activation = 'relu')
            self.cnn_model_structure = max_pool_2d(self.cnn_model_structure, 5)

            self.cnn_model_structure = conv_2d(self.cnn_model_structure, 32, 5, activation = 'relu')
            self.cnn_model_structure = max_pool_2d(self.cnn_model_structure, 5)

            self.cnn_model_structure = fully_connected(self.cnn_model_structure, 1024, activation = 'relu')
            self.cnn_model_structure = dropout(self.cnn_model_structure, 0.8)

            self.cnn_model_structure = fully_connected(self.cnn_model_structure, 2, activation = 'softmax')
            self.cnn_model_structure = regression(self.cnn_model_structure, optimizer = 'adam', learning_rate = model_learning_rate, loss = 'categorical_crossentropy', name = 'targets')

        except:
            print(f'The {BinaryClassifier.define_model.__name__} method could not define the structure of the cnn model.')

        


    def fit_model(self):  

        '''Trains a convolutional neural network model and outputs the trained model'''    

        try:
            self.cnn_model = DNN(self.cnn_model_structure, tensorboard_dir = 'log')
            self.cnn_model.fit({'input': self.X_train}, {'targets': self.y_train}, n_epoch = 3, validation_set = ({'input': self.X_test}, {'targets': self.y_test}), snapshot_step = 500, show_metric = True, run_id = 'anyname')

        except:
            print(f'The {BinaryClassifier.fit_model.__name__} method could not fit the cnn model to the data.')




    def make_prediction(self, image_path):

        '''Input is the path of a single jpg image previously unseen by the convolutional neural network model. 
           Output is the jpg image labeled with its predicted image class as predicted by the cnn model.'''

        try:
            unseen_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
            unseen_image = cv2.resize(unseen_image, (self.image_size, self.image_size))
            no_unseen_images = 1 
            X__unseen_image_array = np.array(unseen_image).reshape(no_unseen_images, self.image_size, self.image_size, self.no_channels) 

            model_prediction = self.cnn_model.predict(X__unseen_image_array)

            if np.argmax(model_prediction) == 1:
                image_class_label = self.image_class1 
            else: 
                image_class_label= self.image_class2

            plt.imshow(unseen_image, cmap='Greys_r')
            plt.title(f'CNN model prediction: {image_class_label}')
            plt.show()
        
        except:
            print(f'The {BinaryClassifier.make_prediction.__name__} method could not predict the class of the image.')

