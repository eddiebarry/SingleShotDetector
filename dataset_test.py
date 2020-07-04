import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from visualiser import save_image, draw_bbox, draw_circle_from_bbox, \
    visualise_img_list
from utils import get_start_and_end_points, get_min_max_white_points
from CONSTANTS import NUM_CLASSES, INPUT_IMAGE_SHAPE, IMGS_PER_CLASS, \
    BATCH_SIZE
from dataset_sequence import SyntheticMNISTDatasetSequence


class DatasetTester():

    # function for testing bbox coords
    def check_bbx_crds(self, image, bbx):
        '''
        Parameters:
            image (numpy array): A numpy array with shape [H,W,C]
            bbx (numpy array): A numpy array with shape [numclasses + 4,] ,
                the contents of which are [P1,P2..Pn,Cx,Cy,w,h]
            
            Refer to this link for more info: 
            https://arxiv.org/pdf/1512.02325.pdf Fig 1

            This function asserts that H==W and the images is in channels
            last format.

            It returns true if the calculated width, height of the non
            black coordinates matches the provided bbx
        '''
        assert image.shape[0] == image.shape[1]
        assert image.shape[2] == 3
        assert bbx.shape[0] >= 4

        x_min, x_max, y_min, y_max = get_min_max_white_points(image)

        calc_start_point = (x_min,y_min)
        calc_end_point   = (x_max,y_max)

        box_start_point, box_end_point = get_start_and_end_points(
            bbx, image.shape)
        
        np.testing.assert_allclose(calc_end_point, box_end_point,1,1)
        np.testing.assert_allclose(calc_start_point, box_start_point,1,1)
        return True


    # Function for creating images and visualising
    def create_images_with_label_and_visualise(self, label, num_visualise=10,
        filepath="./test_results/vis_label_img.png",
        save=False):
        '''
        Parameters:
            label (int): value which denotes which digit images are used
            num_visualise(int): number of images which are to be drawn

            This function uses the dataset to generate num_visualise images and saves 
            all of them in a single image
        '''
        dataset = SyntheticMNISTDatasetSequence(use_one_label=label)
        
        imgs = dataset[0][0].numpy()
        img_list = [imgs[x] for x in range(imgs.shape[0])][:num_visualise]


        visualise_img_list(img_list,filepath=filepath,save=save)
        assert len(img_list) == num_visualise, 'generator failed to gracefully\
            return the correct number of images, returned '+str(len(img_list))\
                +' imgs'


    # Function for creating random images and visualising
    def create_images_and_visualise(self, num_visualise=10,
        filepath="./test_results/vis_all_label_img.png",
        save=False):
        '''
        Parameters:
            num_visualise (int): number of images which are to be drawn

            This function uses the dataset to generate "num_visualise"/NUM_CLASSES
            images for every label and saves all of them in a single image
        '''
        dataset = SyntheticMNISTDatasetSequence()
        img_list = []

        for x in dataset._labels:
            dataset.set_use_single_label(x)
            imgs = dataset[0][0].numpy()
            img_ = [imgs[x] for x in range(imgs.shape[0])][:num_visualise]
            img_list += img_

        visualise_img_list(img_list,filepath=filepath,save=save)
        assert len(img_list) == num_visualise*NUM_CLASSES, 'generator failed to \
            gracefully return the correct number of images, returned '\
                +str(len(img_list)) +' imgs'


    # Function for checking all images are correct dimensions
    def check_dims(self, num_imgs=20):
        '''
        Parameters:
            num_imgs (int): number of images which are to be drawn

            This function uses the dataset to generate "num_imgs" images
            and then checks that the dimensions are the same as the provided
            bbox
        '''
        dataset = SyntheticMNISTDatasetSequence(use_one_label=0)

        for x in range(NUM_CLASSES):
            img = dataset[0][0][0]
            assert img.shape ==  INPUT_IMAGE_SHAPE, "Generated image is\
                not the correct shape " + str(img.shape)


    # def check iterator
    def check_iterator(self,num_imgs=20):
        '''
        Parameters:
            num_imgs (int): number of images which are to be generated

            This function uses the dataset to generate "num_imgs" images,
            checks the dims and sees if the generated coords are the same
            as the image
        '''
        dataset = SyntheticMNISTDatasetSequence()
        
        for x in range(NUM_CLASSES):
            imgs, bboxs = dataset[0]
            img, bbox = imgs[0], bboxs[0]

            assert img.shape ==  INPUT_IMAGE_SHAPE, "Generated image is\
                not the correct shape " + str(img.shape)
            self.check_bbx_crds(img, bbox)


    
if __name__ == "__main__":
    bbx = np.array([0.1,0.9,0.0,  10,10,5,5]) / 300
    img = np.zeros([300,300,3])
    bbox_img = draw_bbox(img, bbx, save=True,
    filepath='./test_results/circle_from_bbx_img_test.jpg')
    
    check = DatasetTester()
    check.check_bbx_crds(bbox_img, bbx)
    print('bbox test passed')

    check.check_iterator()
    print('generator check passed')

    check.create_images_with_label_and_visualise(9,num_visualise=10,
        save=True)
    print('unique label test passed')

    check.create_images_and_visualise(num_visualise=10,save=True)
    print('all label test passed')

    check.check_dims(num_imgs=10)
    print('dimension check passed')