import tensorflow as tf
from model_utils import inverted_res_block, make_divisible, correct_pad, \
    predict_bboxes, get_grid, custom_inv_res_block, Relu6_Conv, ssd_head
from tensorflow.keras import layers
from tensorflow.python.keras.applications import imagenet_utils
from loss import CustomSSDLoss
from CONSTANTS import NUM_CLASSES, INPUT_IMAGE_SHAPE, IMGS_PER_CLASS, \
    SSD_LAYER_NAMES, SSD_SCALES, MOBILENET_WEIGHT_PATH, SSD_LAYER_RATIOS



class SingleShotDetector():
    '''
        Uses the keras implementation of MBILENET V2, 
        extracts features from output layers havings dims

        38, 19, 10 , 5, 3 ,1

        appends feature layers to obtain bbox preds from an
        image

        Just like the paper, it generates 8732 boxes
    '''
    def __init__(self,
        input_shape=INPUT_IMAGE_SHAPE,
        layer_names=SSD_LAYER_NAMES,
        layer_scales=SSD_SCALES,
        layer_ratios=SSD_LAYER_RATIOS,
        model_path=MOBILENET_WEIGHT_PATH
        ):

        self.layer_names = layer_names
        self.layer_scales = layer_scales
        self.layer_ratios = layer_ratios
        self.model, self.grid = self.model_define(input_shape=input_shape, include_top=False, \
            model_path=model_path)
        
        self.model.summary()

    def get_model(self):
        ''' This function exposes the model to the training script'''
        return self.model

    def get_grid(self):
        ''' This function exposes the grid to the training script'''
        return self.grid

    def model_define(self, input_shape=None, alpha=1.0, include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation='softmax',
                model_path=MOBILENET_WEIGHT_PATH):
        ''' 
            Uses keras mobilnet definition for extracting features
            Return the corresponding model grids as well uses the
            inverted resblock modules used in MobileNetv2

            It returns the ssdlite model along with the generated grid
            offsets.
        '''
        # tf.compat.v1.disable_eager_execution()
        input_shape = input_shape

        channel_axis = -1
        img_input = layers.Input(shape=input_shape)

        first_block_filters = make_divisible(32 * alpha, 8)
        x = layers.ZeroPadding2D(
            padding=correct_pad(img_input, 3),
            name='Conv1_pad')(img_input)
        x = layers.Conv2D(
            first_block_filters,
            kernel_size=3,
            strides=(2, 2),
            padding='valid',
            use_bias=False,
            name='Conv1')(
                x)
        x = layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(
                x)
        x = layers.ReLU(6., name='Conv1_relu')(x)

        x = inverted_res_block(
            x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

        x = inverted_res_block(
            x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
        x = inverted_res_block(
            x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

        x = inverted_res_block(
            x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
        x = inverted_res_block(
            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
        x = inverted_res_block(
            x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

        x = inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6)
        x = inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
        x = inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
        x = inverted_res_block(
            x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

        x = inverted_res_block(
            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
        x = inverted_res_block(
            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
        x = inverted_res_block(
            x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

        x, conv_1_out = custom_inv_res_block(
            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=13)
        x = inverted_res_block(
            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
        x = inverted_res_block(
            x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

        x = inverted_res_block(
            x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)


        x, conv_2_out = Relu6_Conv(x,1280,17)
        x, conv_3_out = ssd_head(x,512,18)
        x, conv_4_out = ssd_head(x,256,19)
        x, conv_5_out = ssd_head(x,128,20)
        x, conv_6_out = ssd_head(x,128,21,3)

        feature_layers = [conv_1_out,conv_2_out,conv_3_out,\
            conv_4_out,conv_5_out,conv_6_out]

        bboxes = predict_bboxes(feature_layers,self.layer_names,\
            self.layer_scales, self.layer_ratios)

        single_shot_detector = tf.keras.Model(inputs=img_input, outputs=bboxes)

        grids = get_grid(feature_layers, self.layer_names, \
            self.layer_scales, self.layer_ratios)

        return single_shot_detector, grids

    
        
if __name__ == "__main__":
    det = SingleShotDetector()