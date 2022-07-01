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
 
 ## Training and Evaluation
 Install required Python packages:</br>
 ```
 pip install -r requirement.txt
 ``` 

 
## Checkpoint Downloading
[Google Drive](https://drive.google.com/drive/folders/15vwSnGGHgMkwLQso09RYvXWg7qg9zqge?usp=sharing)

## Cite
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
