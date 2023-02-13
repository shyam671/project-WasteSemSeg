# Project-WasteSemSeg
This respository is a starter-code setup for Resource Constraint Recyclable Waste Segmentation project. It provides the code base for training ENet on ReSort dataset for binary class segmentation.

## Usage

### Data Prepration
* Download the [ResortIT dataset.](https://drive.google.com/file/d/14ThGc53okYC61AnTXFAofiYYY8PTZYtl/view?usp=share_link).
* Unzip the ```dataset.zip``` into the project folder.
* Modify the roor path of the dataset ```__C.DATA.DATA_PATH```.

### Training
* Use ```python train.py``` command to train the model. 
* Change the loss function, model definition, validate funtion (train.py), dataloader(for multiclassicaion)

