# Multi-Task-Learning-Improves-Synthetic-Speech-Detection
This code is for our accepted [manuscript](https://ieeexplore.ieee.org/abstract/document/9746059) to 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

## Data Preprocessing
Downloading the [ASVspoof 2019 Logic Access](https://www.asvspoof.org/database) Dataset.</br>
[Matlab](https://ww2.mathworks.cn/products/matlab.html) is required.</br>
Edit *./extract_feature/process_LA_data.m* according to the absolute path of the dataset.
 ```
 cd extract_feature
 CUDA_VISIBLE_DEVICES=0 /data/users/yangli/Matlab/bin/matlab -nodesktop -nosplash -r process_LA_data.m
 cd ..
 python reload_data.py
 ``` 
 
 ## Training
 Install required Python packages:</br>
 ```
 pip install -r requirement.txt
 ``` 
 Use our multi-task learning methods to train a synthetic speech detection network (Taking oc-softmax loss function as an example.).</br>
 **vanilla**
 ```
 CUDA_VISIBLE_DEVICES=0,1 python3 train.py --add_loss ocsoftmax -o  ./models/ocsoftmax_vanilla -f /data/users/yangli/AIR-ASVspoof-master/LAfeatures/
 ```
 **+bonafide speech reconstruction**
  ```
 CUDA_VISIBLE_DEVICES=0,1 python3 train.py --add_loss ocsoftmax -o  ./models/ocsoftmax_recon_04 -f /data/users/yangli/AIR-ASVspoof-master/LAfeatures/ --S1 --lambda_r 0.04
 ```
 **+spoofing voice conversion**
  ```
 CUDA_VISIBLE_DEVICES=0,1 python3 train.py --add_loss ocsoftmax -o  ./models/ocsoftmax_conver_0003 -f /data/users/yangli/AIR-ASVspoof-master/LAfeatures/ --S2 --lambda_c 0.0003
 ```
 **+speaker classification**
  ```
 CUDA_VISIBLE_DEVICES=0,1 python3 train.py --add_loss ocsoftmax -o  ./models/ocsoftmax_class_00005 -f /data/users/yangli/AIR-ASVspoof-master/LAfeatures/ --S3 --lambda_m 0.00005
 ```
 **Combining all auxillary subtasks**
 ```
 CUDA_VISIBLE_DEVICES=0,1 python3 train.py --add_loss ocsoftmax -o  ./models/ocsoftmax_class_00005_recon_04_conver_0003 -f /data/users/yangli/AIR-ASVspoof-master/LAfeatures/ --S3 --lambda_m 0.00005 --S1 --lambda_r 0.04 --S2 --lambda_c 0.0003
 ```
 
 ## Evaluation
 An example: 
 ```
 CUDA_VISIBLE_DEVICES=0,1 python3 test.py -m ./models/ocsoftmax_class_00005_recon_04_conver_0003  -l ocsoftmax --gpu 0 -f /data/users/yangli/AIR-ASVspoof-master/LAfeatures/
 ```
  For details, please refer to [test.py](https://github.com/mo666666/Multi-Task-Learning-Improves-Synthetic-Speech-Detection/blob/main/test.py).

 
## Checkpoint Downloading
[Google Drive](https://drive.google.com/drive/folders/15vwSnGGHgMkwLQso09RYvXWg7qg9zqge?usp=sharing)

## Cite this
```
@inproceedings{mo2022multi,
  title={Multi-Task Learning Improves Synthetic Speech Detection},
  author={Mo, Yichuan and Wang, Shilin},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6392--6396},
  year={2022},
  organization={IEEE}
}
```
