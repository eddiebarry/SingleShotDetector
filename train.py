import itertools
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from models import SingleShotDetector
from loss import CustomSSDLoss, CustomSSDLossCrossEntropy
from dataset_sequence import SyntheticMNISTDatasetSequence
from visualiser import draw_bbox, draw_pred, visualise_batch
from utils import non_max_suppression
from CONSTANTS import NUM_CLASSES, IMGS_PER_CLASS, INPUT_IMAGE_SHAPE, \
    BATCH_SIZE


class CustomVisCallback(tf.keras.callbacks.Callback):
    '''
        parameters:
            data (data generator class): an instance of the random
                data generator

            grid (tensor): The set of precalculated offsets, which 
                only need to be calculated once

            A callback that visualises model preds at the begining of
            every epoch
        
    '''
    def __init__(self, data, grid, non_max_boxes, \
        out_folder='./test_results/out_img/'):
        super(CustomVisCallback, self).__init__()
        self.data = data
        self.grid = grid
        self.nms_box = non_max_boxes
        self.epoch_num = 0
        self.out_folder = out_folder

    def on_epoch_begin(self,epoch,logs=None):
        print('visualising')
        imgs, bboxes = self.data.__getitem__(0)
        visualise_batch(imgs[:5], bboxes[:5], \
            self.model, self.grid, out_folder=self.out_folder, \
            non_max_boxes=self.nms_box, epoch_num=epoch)
        print('images done | ', epoch, ' : is the current epoch | ',\
            self.nms_box, " : is the nms box")
        if (epoch+1) % 2 == 0:
            self.nms_box = self.nms_box//2
        if self.nms_box < 5:
            self.nms_box = 5


def train_model(data, ssd, loss, optimizers, grid, resume_path=None, \
    epochs=10, steps_per_epoch = NUM_CLASSES*IMGS_PER_CLASS):
    ''' generates a fresh batch and trains'''

    if resume_path is not None:
        ssd.load_weights(resume_path)
        print('model loaded')
    else:
        resume_path = "./weights/seq_{epoch:02d}-{loss:.2f}.hdf5"

    model_checkpoint_callback = \
        tf.keras.callbacks.ModelCheckpoint(filepath=resume_path, \
        save_best_only=True, monitor='loss')
    lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(patience=5,\
        monitor='loss')
    custom_vis_callback = CustomVisCallback(data,grid,1000)

    ssd.compile(optimizer=optimizers,loss = loss)

    history = ssd.fit( data, epochs = epochs, \
        callbacks=[lr_on_plateau,model_checkpoint_callback,\
            custom_vis_callback])

if __name__ == "__main__":
    custom_datas = SyntheticMNISTDatasetSequence(batch_size=BATCH_SIZE)
    model_class = SingleShotDetector()

    single_shot_detector = model_class.get_model()
    detector_grid = model_class.get_grid()

    criterion = CustomSSDLoss(grid=detector_grid, ratio=3.0)

    imgs, bboxes = custom_datas[0]
    visualise_batch(imgs, bboxes, \
        single_shot_detector, detector_grid, \
            out_folder="./test_results/model_output/", non_max_boxes=5, 
            epoch_num=0)

    train_model(data=custom_datas, ssd=single_shot_detector, loss=criterion, \
        optimizers=tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=0.9), \
            epochs=30,grid=detector_grid)