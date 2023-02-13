# Project-WasteSemSeg
This respository is a starter-code setup for Resource Constraint Recyclable Waste Segmentation project. It provides the code base for training ENet on ReSort dataset for binary class segmentation.

## Usage

### Data Prepration
* Download the [ResortIT dataset.](https://drive.google.com/file/d/14ThGc53okYC61AnTXFAofiYYY8PTZYtl/view?usp=share_link).
* Unzip the ```dataset.zip``` into the project folder.
* Modify the root path of the dataset by changing ```__C.DATA.DATA_PATH``` in ```config.py```.

### Training
* Use ```python train.py``` command to train the model.
* ```train.py``` also provides the flexibility of either training entire model (encoder + decoder) or just encoder which can be performed by changing ```__C.TRAIN.STAGE``` in ```config.py```.
* To Do
  - For Instance Segmentation the training loss needs to be modified from Binary Cross Entropy.
  - ```model.py``` contains the model definition of ENet. To train on newer models such as ICNet model definition of such models needs be added in```model.py```.
  - Changing from Binary Segmentation to Instance Segmentation the ```validate``` function of ```train.py``` and dataloader class ```resortit``` needs to be modified accordingly.
  - Scripts to calculate ```FLOPS``` and  ```# of trainable model parameters```.

