# VTT video emotion recognition

Video Emotion Recognition for VTT Friends dataset, trained with RAF dataset and AffectNet dataset

![Alt text](example.png)

- - -

### Preliminaries
#### Packages
* python==2.7.1
* pytorch==1.3.0
* numpy==1.16.5
* opencv-python==4.1.0
* matplotlib==2.2.4
* face_recognition==1.2.3

#### Datasets
* AffectNet dataset [#](http://mohammadmahoor.com/affectnet/)
* RAF dataset [#](http://www.whdeng.cn/RAF/model1.html)
* VTT Friends dataset

#### Preprocessing
* Parsed AffectNet dataset can be preprocessed by using `parse_AffectNet.py` script
    * Parsed AffectNet dictionary has a categorical structure of [af_dict] -> [emotion label] -> [example index] -> ['img':image filenames], ['gt':ground truth bounding boxes], ['emo':emotion labels], ['aro':arousal intensity]
* Parsed RAF dataset can be preprocessed by using `parse_RAF.py` script
    * Parsed RAF dictionary has a categorical structure of [raf_dict] -> [example index] -> ['img':image filenames], ['gt':ground truth bounding boxes], ['em':emotion labels]
    
* VTT Friends dataset (episode 1-10) is used as the validation set
    * All frames of the videos should be extracted and saved as image files as in `extract_friends.py`
    * Metadata (json) files are used for processing as in `parse_friends_new.ipynb` to extract facial region images and emotion labels
    * The dataset can be indexed using `friends_parsed_new.npy`, with the structure of [val_dict] -> [emotion label] -> [example index] -> ['img':image filenames], ['pos':episode num, character id], ['emo':emotion label]


### Train
`model_train.py` and `model_tsm_train.py` are training script where training variables are configured in `ops.py`. Training can be performed by running
```
python model_train.py
```
where the training process (error and accuracy) can be plotted by running,
```
python plot_errval.py
```

### Test
Testing for a video frame can be performed by importing the network model function by,
```
from resnet_tsm import resnet18 as resnet
```
where the model function is resnet(input_img), input_img is cropped RGB face sequence image (4x3x224x224), where 4 is the buffer size for video sequence.

#### Results
Results for VTT_friends dataset can be obtained using `create_data.ipynb` and stored as in `friends_s01_ep00.json` as,
```
{"type": "emotion", "class": "happy", "seconds": 15.0, "object": Object}, 
...
```

- - -

#### References
* Cohn-Kanade (CK+) dataset
    - Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
    - Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

* AffectNet dataset
    - Ali Mollahosseini, Behzad Hasani, and Mohammad H. Mahoor, “AffectNet: A New Database for Facial Expression, Valence, and Arousal Computation in the Wild”, IEEE Transactions on Affective Computing, 2017.
    
* RAF dataset
    - Li, Shan and Deng, Weihong and Du, JunPing, "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild", CVPR 2017

#### Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
