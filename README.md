# Single Shot detector

#### This project implements the ideas presented in 

https://arxiv.org/pdf/1512.02325.pdf

https://arxiv.org/pdf/1801.04381.pdf

---

For a quick dive into the project, run

```
https://github.com/eddiebarry/SingleShotDetector/blob/master/SSDLite.ipynb
```

in colab

---

#### Code Structure

- weights

  - best_weight.hdf5

    ###### All model weights generated during training are stored here

- test_results

  - model_output 

    ###### Results of model inference are stored here

  - out_img

    ###### Visualisations during model training are stored here

  - inp_img

    ###### Model inputs which are fed to the model before predictions are stored here

- CONSTANTS.py

  ###### Model configuration options such as BATCH_SIZE, NUM_CLASSES, NUM_IMAGES_PER_CLASS are stored here

- dataset_sequence.py

  ###### This file is responsible for wrapping the grabbing digits from MNIST data, scaling them and appending them to blank images. Because all images are generated online, no image is seen twice. and because of random scaling and appending, data augmentation is made redundant

- dataset_test.py

  ###### Tests written to ensure that changes in the dataset generation do not change the model inputs

- models.py

  ###### This file contains the high level definition of the MobileNet SSDLite architecture

- visualiser.py

  ###### This file contains the code relevant to visualising images, image lists, model predictions. After drawing bboxes, the images are scaled to (800,800) for easier visualisation

- loss.py

  ###### This file contains the implementation of 2 losses. CustomSSDLoss which is the loss function presented in the SSD paper, and CustomSSDLossCrossEntropy which uses cross entropy instead of softmax loss for class prediction. 

  ###### Empirical experimentation found that the paper's loss was better and hence is the default set in the training script. 

  ###### The loss uses the grid offsets/anchors generated by the model during initialisation so that the expensive operation is not needlessly repeated

- train.py

  ###### This file contains the model.fit method along with the model prediction visualisation callback which saves the image predictions at the start of every epoch

- test.py

  ###### This file contains code for model visualisation

- Finetune.py

  ###### This file contains code for finetuning the model on larger images. Empirical evidence indicates that one iteration is enough for good results

- img_utils.py

  ###### Contains neccessary functions for processing images

- model_utils.py

  ###### Contains functions for extracting preds from the model architecture as well as generating a grid the first time

- utils.py

  ###### Contains misc utils suchs as NMS code. In NMS, we disregard background predictions and perform NMS only on predictions which are not labeled as the background

---

### Dependencies

```
tensorflow=2.2.0

opencv-python

numpy

matplotlib
```

----

### Visualising Dataset

##### Inorder to visualise the created images, run

```python
python dataset_test.py
```

A visualisation of generated images are saved in

```
'./test_results/vis_all_label_img.png
'./test_results/vis_label_img.png'
```

---

### Training

##### Inorder to start training on the generated dataset, run

```
python train.py
```

At the start of each epoch, model predictions are visualised and saved in 

```
'./test_results/out_img/' 
```

by default

---

### Testing

##### To run the predictions, run

```
python test.py
```

All generated images are stored in

```
'./test_results/model_outputs/'
```