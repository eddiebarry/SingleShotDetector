import numpy as np

NUM_CLASSES=10
NUM_PREDS_PER_BOX = NUM_CLASSES+1 + 4
INPUT_IMAGE_SHAPE = (300,300,3)
IMGS_PER_CLASS = 1000

BATCH_SIZE = 32

SSD_LAYER_NAMES = [
    '1',  #38*38 - 4 boxes
    '2', #19*19 - 6 boxes
    '3', #10*10 - 6 boxes
    '4', #5*5   - 6 boxes
    '5', #3*3   - 4 boxes
    '6', #3*3   - 4 boxes
]
ssd_layer_ratios_temp = {
    '1':[1,2,0.5],          #38*38 - 4 boxes
    '2':[1,2,3,0.5,1/3.0], #19*19 - 6 boxes
    '3':[1,2,3,0.5,1/3.0], #10*10 - 6 boxes
    '4':[1,2,3,0.5,1/3.0], #5*5   - 6 boxes
    '5':[1,2,0.5],         #3*3   - 4 boxes
    '6':[1,2,0.5],         #1*1   - 4 boxes
}

SCALE_MIN = 0.2
SCALE_MAX = 0.9

# from paper formula 4
SSD_SCALES = [ SCALE_MIN + ((SCALE_MAX-SCALE_MIN)/ \
    (len(SSD_LAYER_NAMES)-1))*(x-1) \
    for x in range(1,len(SSD_LAYER_NAMES)+2,1)]

# adding scale 1 box which is sqrt of product of two scales
for idx,key in enumerate(ssd_layer_ratios_temp.keys()):
    ssd_layer_ratios_temp[key].append( np.sqrt( SSD_SCALES[idx]*SSD_SCALES[idx+1] ) )

SSD_LAYER_RATIOS = ssd_layer_ratios_temp

MOBILENET_WEIGHT_PATH = "./weights/mobilenet_v2_weights_tf_dim" \
    +"_ordering_tf_kernels_1.0_224_no_top.h5"