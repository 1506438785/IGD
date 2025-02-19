# IGD_Vehicle_Exterior_Shape

## Authors:
- Yuhao Liu  
- Maolin Yang  
- Pingyu Jiang  
- All authors are affiliated with the State Key Laboratory of 
- Mechanical Manufacturing Systems at Xi'an Jiaotong University.
---

## Code Usage Instructions:

### 1. Python Version:
Python 3.10  

### 2. Install Required Libraries:
Install the required libraries using the following command:   

PyTorch library should be selected from the official website 
according to your computer's CUDA version.

```bash
pip install -r requirements.txt
```
### 3. Folder Structure and Contents:
- The project contains the following folders and files:
- a) models_3D_obj folder:
- Vehicle exterior 3D dataset in .obj format.
- https://drive.google.com/file/d/1ycDRq9_Oflg1d-6r7j4Fn-as5WTvc_RR/view?usp=drive_link
- b) models_labels_npy folder:
- Vehicle label files, input conditions processed as 2048x3 in .npy format.
- c) model_pointcloud_npy folder:
- Vehicle point cloud data, sampled at 2048x3 in .npy format.

### 4. Scripts:
a) Training Script:
Run the following command to train the model:
```bash
python run train_improved_cgan.py
```
b) Conditional Generation Script:
Run the following command for conditional generation:
```bash
python run test_improved_cgan.py
```