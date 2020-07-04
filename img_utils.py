import cv2
import numpy as np
from utils import get_min_max_white_points


# function for stripping mnist image of blackspace
# not vectorised because different images have different crops
def strip_blackspace(images):
    '''
    Parameters:
        images (numpy array): A numpy array with shape [N,H,W,C]

        This function used the data in bbx to get top left and bottom 
        right corners of an image which are used to crop the digits out 
        of the image
    '''
    cropped_imgs = []
    for img in images:
        x_min, x_max, y_min, y_max = get_min_max_white_points(img)
        cropped_img = img[y_min:y_max,x_min:x_max]
        cropped_imgs.append(cropped_img)

    return cropped_imgs


# function for scaling and appending images as random positions
def scale_and_append_images(images,tensor_shape,scale=[0.1,0.9],\
    rand_type='uniform'):
    '''
    Parameters:
        images (numpy array): A numpy array with shape [N,H,W,C]
        scale (numpy array): A numpy array containing [high, low]
            where scale is selected randomly

        This function scales the images while preserving aspect ratios,
        finds a appropriate new x,y coordinate and appends it to 
        a black image at that position
    '''
    if rand_type == 'uniform':
        scales = np.random.uniform(low=scale[0], high=scale[1], \
            size=len(images))
    
    img_default = np.zeros(tensor_shape)
    scaled_and_appended_imgs = []

    for img, scale in zip(images,scales):
        img_ = np.copy(img_default)

        flag, rescaled_img = scale_image(img, scale,tensor_shape)
        if not flag:
            continue
        
        new_pos = new_random_position(rescaled_img,tensor_shape)
        img_[new_pos[0]:new_pos[1],new_pos[2]:new_pos[3],0]=rescaled_img
        img_[new_pos[0]:new_pos[1],new_pos[2]:new_pos[3],1]=rescaled_img
        img_[new_pos[0]:new_pos[1],new_pos[2]:new_pos[3],2]=rescaled_img
        
        scaled_and_appended_imgs.append(img_)
    
    #case where all images are corrupted
    if len(scaled_and_appended_imgs)==0:
        return [None]

    return scaled_and_appended_imgs


# function for scaling image and preserving aspect ratio
def scale_image(image, scale, tensor_shape):
    '''
    Parameters:
        images (numpy array): A numpy array with shape [H,W,C]
        scale (float): A float indicating the fraction that
            should be occupied by the new images largest side

        This function scales the images while preserving aspect ratios,
        and returns it
    '''
    # scale horizontaly
    if image.shape[0]>image.shape[1]:
        new_shape = (scale*tensor_shape[0], 
            image.shape[1]*scale*tensor_shape[0]/image.shape[0])
        new_shape =  tuple(map(int,new_shape))
    else:
        new_shape = (image.shape[0]*scale*tensor_shape[1]/\
            image.shape[1], scale*tensor_shape[1])
        new_shape =  tuple(map(int,new_shape))
    
    try:
        new_image = cv2.resize(image,new_shape)
    except Exception as e:
        # print(e)
        # print(image.shape)
        # print(new_shape)
        return False, 1

    return True, new_image


# function for returning a random position according to image scale
def new_random_position(image, tensor_shape, rand_type='uniform'):
    '''
    Parameters:
        images (numpy array): A numpy array with shape [H,W,C]

        This function checks all possible places where the new image
        can be appended and returns a random point
    '''
    low = (0,0)
    high = tensor_shape[0]-image.shape[0], \
        tensor_shape[1]-image.shape[1]
    
    if rand_type == 'uniform':
        new_x = int(np.random.uniform(low[0], high[0], 1))
        new_y = int(np.random.uniform(low[1], high[1], 1))

    return (new_x, new_x+image.shape[0],new_y, new_y+image.shape[1], )


# function for splitting mnist according to labels:
def get_class_images(images, labels, num_class):
    '''
    Parameters:
        images (numpy array): All the images in the dataset
            joined along axis 0 such that the shapes is [N, H, W, C]
        labels (numpy array): The labels of the images such that 
            label[i] correspons to image[i]
        
    asserts that the num classes initialised is the same as
    the number of classes present in the labels provided

    This function converts the labels and images into a dict where each
    label corresponds to all images in the dataset
    '''
    assert num_class == len(np.unique(labels)), 'all labels not\
        present in dataset'
    
    img_dict = {}
    for lab in np.unique(labels):
        idxs = np.argwhere(labels==lab)
        img_dict[lab] = np.transpose(images[idxs],axes=[0,2,3,1])

    return img_dict