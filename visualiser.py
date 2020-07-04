import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import non_max_suppression


def save_image(image,filepath='./test_results/vis_img.jpg'):
    '''
        Parameters:
            image (numpy array): A numpy array with shape [H,W,C]

            This functions saves the image as a file
    '''
    cv2.imwrite(filepath,image)


def draw_bbox(image,bbx,filepath='./test_results/bbx_img.jpg',save=False):
    '''
        Parameters:
            image (numpy array): A numpy array with shape [H,W,C]
            bbx (numpy array): A numpy array with shape [numclasses + 4,] , 
            the contents of which are [P1,P2..Pn,Cx,Cy,w,h]

            This functions saves the image as a file after drawing the bbx
    '''
    image = image.copy()
    
    start_point = bbx[-4] - (bbx[-2]/2.0) , bbx[-3] - (bbx[-1]/2.0)
    end_point   = bbx[-4] + (bbx[-2]/2.0) , bbx[-3] + (bbx[-1]/2.0)

    start_point = start_point[0]*image.shape[0], \
                    start_point[1]*image.shape[1]
    end_point = end_point[0]*image.shape[0], \
                    end_point[1]*image.shape[1]

    start_point = tuple(map(int,start_point))
    end_point   = tuple(map(int,  end_point))
    
    image = cv2.rectangle(image, start_point, end_point, 255 )

    if save:
        cv2.imwrite(filepath,image)
    
    return image

def draw_pred(image,bbxs,filepath='./test_results/bbx_img.jpg',save=False,\
    image_shape=(800,800)):
    '''
        Parameters:
            image (numpy array): A numpy array with shape [H,W,C]
            bbx (numpy array): A numpy array with shape [numclasses + 4,] , 
            the contents of which are [P1,P2..Pn,Cx,Cy,w,h]

            This functions saves the image as a file after drawing the bbx
    '''
    image = image.copy()
    
    for idx in range(bbxs.shape[0]):
        bbx = bbxs[idx]
        # start_point = bbx[-4] - (bbx[-2]/2.0) , bbx[-3] - (bbx[-1]/2.0)
        # end_point   = bbx[-4] + (bbx[-2]/2.0) , bbx[-3] + (bbx[-1]/2.0)

        start_point = bbx[1]*image_shape[0], \
                        bbx[0]*image_shape[1]
        end_point = bbx[3]*image_shape[0], \
                        bbx[2]*image_shape[1]

        start_point = tuple(map(int,start_point))
        end_point   = tuple(map(int,  end_point))

        

        if int(bbx[4])==10:
            color = (0,0,255)
        else:
            color = 255
        
        image = cv2.resize(image,image_shape)

        image = cv2.putText(image,\
            "label : " +str(bbx[4].numpy()),\
            start_point, cv2.FONT_HERSHEY_SIMPLEX , 1 , color,\
            1, cv2.LINE_AA)

        image = cv2.putText(image, "conf : "+str(bbx[5].numpy()),\
            (start_point[0],start_point[1]+30), \
            cv2.FONT_HERSHEY_SIMPLEX , 1 , color, 1, cv2.LINE_AA)

        image = cv2.rectangle(image, start_point, end_point, color )
        

    if save:
        cv2.imwrite(filepath,image)
    
    return image

def draw_circle_from_bbox(image,bbx,
    filepath='./test_results/circle_from_bbx_img.jpg',save=False):
    '''
        Parameters:
            image (numpy array): A numpy array with shape [H,W,C]
            bbx (numpy array): A numpy array with shape [numclasses + 4,] , 
            the contents of which are [P1,P2..Pn,Cx,Cy,w,h]

            This functions saves the image as a file after drawing the bbx
    '''
    image = image.copy()

    start_point = bbx[-4] , bbx[-3]
    radius   = (bbx[-1] + bbx[-2]) / 2.0

    start_point = start_point[0]*image.shape[0], \
                    start_point[1]*image.shape[1]
    radius = radius*image.shape[0]
                    
    start_point = tuple(map(int,start_point))
    radius      = int(radius)

    image = cv2.circle(image,start_point, radius, 255 )

    if save:
        cv2.imwrite(filepath,image)
    
    return image

def visualise_img_list(img_list,
    filepath="./test_results/vis_img_list.png",
    save=False,
    label='default_label'):
    '''
        Parameters:
            img_list (numpy array): A numpy array with shape [N,H,W,C]

            This function saves all of the images in a single image
    '''
    n = np.ceil(np.sqrt(len(img_list)))
    m = np.ceil(len(img_list)/n)
    plt.figure(figsize=(3*n,3*m))
    for i, img in enumerate(img_list):
        plt.subplot(m,n,i+1)
        plt.xticks([]), plt.yticks([])
        plt.title(label)
        if img.shape[2]==1:
            plt.imshow(np.squeeze(img,axis=2)/255,cmap='gray')
        else:
            plt.imshow(img/255)
    plt.tight_layout()
    
    if save:
        plt.savefig(filepath)

def visualise_batch(imgs, bboxes, ssd, grid, epoch_num=0, \
    out_folder='./test_results/out/', non_max_boxes=1000):
    '''
        Parameters :
            imgs (list of numpy images):  A list of numpy images arrays
            bboxes (list of numpy arrays): For every corresponding image,
            a corresponding list of bboxes


            Takes a batch of images and bboxes,
            Saves the input images by default in a folder called 
            "./test_results/inp_img/"

            Performs NMS on the predicted boxes and saves the images with
            overlaid boxes in "./test_results/out_img/"
    '''
    for idx, x in enumerate(zip(imgs,bboxes)):
        draw_bbox(x[0].numpy(),x[1].numpy(),\
            filepath='./test_results/inp/'\
                +str(epoch_num)+'_' +str(idx)+ '.jpg',save=True)
                
    predictions = ssd.predict(imgs)

    new_cx = np.expand_dims(\
        (predictions[:,:,11]*grid[:,:,13])+grid[:,:,11],axis=2)
    new_cy = np.expand_dims(\
        (predictions[:,:,12]*grid[:,:,14])+grid[:,:,12],axis=2)
    new_w  = tf.expand_dims(\
        tf.exp(predictions[:,:,13])*grid[:,:,13],axis=2)
    new_h  = tf.expand_dims(\
        tf.exp(predictions[:,:,14])*grid[:,:,14],axis=2)
    
    new_boxes = tf.concat( \
        [ predictions[:,:,:11] , new_cx, new_cy, new_w, new_h ] \
            ,axis=2)

    for idx in range(imgs.shape[0]):
        bboxes = non_max_suppression(new_boxes[idx].numpy(), \
            non_max_boxes=non_max_boxes)
        img = imgs[idx].numpy()

        draw_pred(img, bboxes, \
            filepath=out_folder +str(epoch_num) \
            +'_' +str(idx)+ '.jpg',save=True)


if __name__ == "__main__":
    
    bbx = np.array([10,10,5,5])/20
    img = np.zeros([20,20,1])
    draw_circle_from_bbox(img,bbx,save=True)
    draw_bbox(img,bbx,save=True)
    
    