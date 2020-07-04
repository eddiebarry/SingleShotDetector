import itertools
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from models import SingleShotDetector
from loss import CustomSSDLoss
from dataset_sequence import SyntheticMNISTDatasetSequence
from visualiser import draw_bbox, draw_pred, visualise_batch
from utils import non_max_suppression, get_bboxes_from_images
from CONSTANTS import NUM_CLASSES, IMGS_PER_CLASS, INPUT_IMAGE_SHAPE,\
    BATCH_SIZE


if __name__ == "__main__":
    data = SyntheticMNISTDatasetSequence()
    model_class = SingleShotDetector()

    single_shot_detector = model_class.get_model()

    weights_path = "./weights/finetune.hdf5"

    single_shot_detector.load_weights(weights_path)
    detector_grid = model_class.get_grid()

    imgs, bboxes = data[0]

    visualise_batch(imgs, bboxes, \
        single_shot_detector, detector_grid, \
            out_folder="./test_results/model_output/", non_max_boxes=1, 
            epoch_num=0)