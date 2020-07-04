import cv2
import random
import itertools
import numpy as np
import tensorflow as tf
from visualiser import save_image, visualise_img_list
from utils import get_bboxes_from_images, get_start_and_end_points, \
    get_min_max_white_points
from img_utils import strip_blackspace, scale_and_append_images, \
    scale_image, new_random_position, get_class_images
from CONSTANTS import NUM_CLASSES, INPUT_IMAGE_SHAPE, IMGS_PER_CLASS, \
    BATCH_SIZE


class SyntheticMNISTDatasetSequence(tf.keras.utils.Sequence):
    '''
        This class is used to strip and append MNIST images on a blank 
        image. It randomly appends the digit on the image and generates
        bboxes and exposes the generation using a python
    '''

    # function for init from constants can be overriden
    def __init__(self, batch_size=BATCH_SIZE, num_class=NUM_CLASSES, \
        img_shape=INPUT_IMAGE_SHAPE,num_imgs_per_class=IMGS_PER_CLASS, \
        use_one_label=None,scales=[0.1,0.9]):
        '''
        Parameters:
            num_class (int): The total number of classes in the dataset
            img_shape (list): The final processed image shape
            num_imgs_per_class (int): The number of image, bbox pairs that
                need to be generated per class 

        This function sets the above values, downloads the MNIST dataset,
        and stores each image according to it's labels
        '''
        self._num_class = num_class
        self._img_shape = img_shape
        self._num_imgs_per_class = num_imgs_per_class
        self._use_single_label = use_one_label
        self._batch_size = batch_size
        self._scale = scales

        (mn_x_train, mn_y_train), (mn_x_test, mn_y_test) = \
            tf.keras.datasets.mnist.load_data(path="mnist.npz")
        
        self._labels = np.unique(np.append(mn_y_train,mn_y_train,axis=0)) 

        # dict for storing each class images
        self._images_data_by_class = get_class_images(
            np.append(mn_x_train,mn_x_test,axis=0),
            np.append(mn_y_train,mn_y_test,axis=0),
            self._num_class
        )

        mn_x_train, mn_y_train, mn_x_test, mn_y_test = None, None, \
            None, None

    def __len__(self):
        return (self._num_class * self._num_imgs_per_class) // \
            self._batch_size

    def __getitem__(self, idx):
        ''' 
            Returns a batch of data of the shape [Batch size, (im_w,im_h,3)]
            and the corresponding bboxes of the form 
            [Batch, [c1,c2,c3,...,cn,Cx,Cy,w,h] ]
        '''
        imgs = []
        boxs = []
        for x in range(self._batch_size):
            img, bbox = self.get_single_img_tensor()
            imgs.append(img)
            boxs.append(bbox)

        return (tf.concat(imgs, axis=0), tf.concat(boxs,axis=0))

    def get_single_img_tensor(self):
        '''
            This function returns 2 tensors of shape (300,300,3) and
            (15,) per image.

            It first strips the blackspace from MNIST images, then
            it randomly selects a position and appends it on the larger 
            images            
        '''
        if self._use_single_label is None:
            label_picked = np.random.choice(self._labels)
        else:
            label_picked = self._use_single_label

        num_elems    = self._images_data_by_class[label_picked].shape[0]
        
        final_image = [None]
        label = [label_picked]
        while final_image[0] is None:
            elem_picked  = int(np.random.uniform(low=0,high=num_elems,size=1))
            rand_img = [self._images_data_by_class[label_picked][elem_picked]]

            stripped_image = strip_blackspace(images=rand_img)

            final_image = scale_and_append_images(stripped_image,\
                tensor_shape=self._img_shape,scale=self._scale)
            if final_image[0] is not None:
                bbox        = get_bboxes_from_images(final_image,label)
        
        final_image = tf.convert_to_tensor(np.array(final_image),dtype=tf.float32)
        bbox = tf.convert_to_tensor(bbox,dtype=tf.float32)

        return final_image, bbox

    # function for setting whether a single label must be generated or not
    def set_use_single_label(self,use_single_label):
        '''
            This function sets the value of use_single_label. 
            if this label is set to None, random images are generated.
            Otherwise images belonging to this particular label are selected
        '''
        self._use_single_label = use_single_label


if __name__ == "__main__":
    dataset = SyntheticMNISTDatasetSequence(batch_size=BATCH_SIZE)
