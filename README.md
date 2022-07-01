# Multi-Task-Learning-Improves-Synthetic-Speech-Detection
This code is for our accepted [manuscript](https://ieeexplore.ieee.org/abstract/document/9746059) to 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

## Data Processing
Downloading the [ASVspoof 2019 Logic Access](https://www.asvspoof.org/database) Dataset.</br>
[Matlab](https://ww2.mathworks.cn/products/matlab.html) for data preprocessing.</br>
Edit *./extract_feature/process_LA_data.m* according to the absolute path of the dataset.
 ```
 cd extract_feature
 CUDA_VISIBLE_DEVICES=0 /data/users/yangli/Matlab/bin/matlab -nodesktop -nosplash -r process_LA_data.m
 ``` 
 
 ## Training and Evaluation
 Install Python packages:</br>
 ```
 pip install -r requirement.txt
 ``` 

 
## Checkpoint Downloading
[Google Drive](https://drive.google.com/drive/folders/15vwSnGGHgMkwLQso09RYvXWg7qg9zqge?usp=sharing)

## Cite
@inproceedings{mo2022multi,</br>
  &emsp;title={Multi-Task Learning Improves Synthetic Speech Detection},</br>
  &emsp;author={Mo, Yichuan and Wang, Shilin},</br>
  &emsp;booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},</br>
  &emsp;pages={6392--6396},</br>
  &emsp;year={2022},</br>
  &emsp;organization={IEEE}</br>
}
