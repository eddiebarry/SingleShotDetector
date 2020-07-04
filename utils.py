import numpy as np
import tensorflow as tf
from CONSTANTS import NUM_CLASSES



def get_start_and_end_points(bbx,image_shape):
    '''
    Parameters:
        bbx (numpy array): A numpy array with shape [numclasses + 4,] ,
            the contents of which are [P1,P2..Pn,Cx,Cy,w,h]
        image_shape : a list of 3 ints containing [H, W, C]

        Refer to this link for more info: 
        https://arxiv.org/pdf/1512.02325.pdf Fig 1


        Using the image shape, it calculates the projected bbox 
        coordinates. returns the top left and bottom right points
        in x,y format
    '''
    start_point = bbx[-4] - (bbx[-2]/2.0) , bbx[-3] - (bbx[-1]/2.0)
    end_point   = bbx[-4] + (bbx[-2]/2.0) , bbx[-3] + (bbx[-1]/2.0)

    start_point = start_point[0]*image_shape[0]-2, \
        start_point[1]*image_shape[1]-2
    end_point = end_point[0]*image_shape[0]+2, \
        end_point[1]*image_shape[1]+2
    
    start_point = tuple(map(int,start_point))
    end_point   = tuple(map(int,  end_point))

    return start_point, end_point

def get_min_max_white_points(image):
    '''
    Parameters:
        image (numpy array): A numpy array with shape [H,W,C]

        Using the image, it calculates a bbox around all the white
        pixels. It then returns min and max X,Y coords.
    '''
    points = np.argwhere(image>0)

    x_min, y_min = np.min(points[::,1]), np.min(points[::,0])
    x_max, y_max = np.max(points[::,1]), np.max(points[::,0])

    return x_min-2, x_max+2, y_min-2, y_max+2

def get_bboxes_from_images(images, labels):
    '''
    Parameters:
        images (numpy array): A numpy array with shape [N,H,W,C]

        Using the images, it calculates a bbox around all the white
        pixels. It then returns a numpy array with shape [None, 4] , 
            the contents of which are [ [c1, c2, c3, ..., Cx,Cy,w,h], ... ]
    '''
    bboxes = np.empty([0,NUM_CLASSES+1+4])
    for image, label in zip(images,labels):
        points = np.argwhere(image>0)

        # import pdb
        # pdb.set_trace()

        x_min, y_min = np.min(points[:,1]), np.min(points[:,0])
        x_max, y_max = np.max(points[:,1]), np.max(points[:,0])

        # Format of BBOX : [Cx, Cy, w, h]
        bbox = np.array([[
            (x_min+x_max)/2/image.shape[0],
            (y_min+y_max)/2/image.shape[1],
            (x_max-x_min)/image.shape[0],
            (y_max-y_min)/image.shape[1],
            ]]
            )
        
        confidences = tf.keras.utils.to_categorical(label, NUM_CLASSES+1)
        confidences = np.expand_dims(confidences,0)
        
        bbox = np.append(confidences,bbox,1)
        bboxes = np.append(bboxes,bbox,axis=0)

    return bboxes

def non_max_suppression(bboxes, non_max_boxes):
    '''
    Parameters:
        bboxes (tensor array) : Tensor array with shape 
            [ NUM_BOXES. [c1,c2,c3,...,cn,Cx,Cy,w,h] ] which is the output
            of ssd model. assumes grid offsets have been already added

        Takes a tensor array of format  
        [ NUM_BOXES. [c1,c2,c3,...,cn,Cx,Cy,w,h] ]
        and uses the confidences to perform Non maximum suppression. Any Bbox
        with iou greater than 0.5 is dropped if it's confidence is lower and
        all background preds are dropped
    '''
    confidences = tf.reduce_max(tf.nn.softmax(bboxes[:,:11]),axis=1)
    labels = tf.argmax(bboxes[:,:11],axis=1)
    mask = 1 - tf.cast(labels==10,tf.int32)
    confidences = confidences*tf.cast(mask,tf.float32)

    
    x_min_grid =  tf.expand_dims(bboxes[:,11] - \
        tf.divide(bboxes[:,13],2.0),axis=1)
    x_max_grid =  tf.expand_dims(bboxes[:,11] + \
        tf.divide(bboxes[:,13],2.0),axis=1)
    y_min_grid =  tf.expand_dims(bboxes[:,12] - \
        tf.divide(bboxes[:,14],2.0),axis=1)
    y_max_grid =  tf.expand_dims(bboxes[:,12] + \
        tf.divide(bboxes[:,14],2.0),axis=1)


    bboxes = tf.concat(\
        [y_min_grid,x_min_grid,y_max_grid,x_max_grid],axis=1)

    suppresed_box_idx = tf.image.non_max_suppression(\
        bboxes, scores=confidences, max_output_size=non_max_boxes, \
        iou_threshold=0.5)

    supp_boxes = tf.gather(bboxes,suppresed_box_idx)
    labels = tf.expand_dims(tf.gather(labels,suppresed_box_idx),axis=1)
    labels = tf.cast(labels,tf.float32)

    conf   = tf.expand_dims(\
        tf.gather(confidences,suppresed_box_idx),axis=1)

    return tf.concat([supp_boxes,labels,conf],axis=1)