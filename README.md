WLASL: A large-scale dataset for Word-Level American Sign Language (WACV 20' Best Paper Honourable Mention)
============================================================================================

This repository contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".

Please **star the repo** to help with the visibility if you find it useful.

Please **star the repo** to help with the visibility if you find it useful.

**yt-dlp vs youtube-dl** youtube-dl has had low maintance for a while now and does not work for some youtube videos, see this [issue](https://github.com/ytdl-org/youtube-dl/issues/30568).
[yt-dlp](https://github.com/yt-dlp/yt-dlp) is a more up to date fork, which seems to work for all youtube videos. Therefore `./start_kit/video_downloader.py` uses yt-dlp by default but can be switched back to youtube-dl in the future by adjusting the `youtube_downloader` variable.
If you have trouble with yt-dlp make sure update to the latest version, as Youtube is constantly changing.

Download Original Videos
-----------------
1. Download repo.
```
git clone https://github.com/sharif110/SilentTalk.git
```

2. Install [youtube-dl](https://github.com/ytdl-org/youtube-dl) for downloading YouTube videos.
3. Download raw videos.
```
cd start_kit
python video_downloader.py
```
4. Extract video samples from raw videos.
```
python preprocess.py
```
5. You should expect to see video samples under directory ```videos/```.

Requesting Missing / Pre-processed Videos
-----------------

Videos can dissapear over time due to expired urls, so you may find the downloaded videos incomplete. In this regard, we provide the following solution for you to have access to missing videos.

We also provide pre-processed videos for the full WLASL dataset on request, which saves troubles of video processing for you.

 (a) Run
```
python find_missing.py
```
to generate text file missing.txt containing missing video IDs.

 (b)  Submit a video request by agreeing to terms of use at:  https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link. You will get links to the missing videos within 7 days (recently I got more occupied, some delays may be expected yet I'll try to share in one week. If you are in urgent need, drop me an email.)


File Description
-----------------
The repository contains following files:

 * `WLASL_vx.x.json`: JSON file including all the data samples.

 * `data_reader.py`: Sample code for loading the dataset.

 * `video_downloader.py`: Sample code demonstrating how to download data samples.

 * `README.md`: this file.


Data Description
-----------------

* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.

* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.

* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.

* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.

* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.

* `instance_id`: *int*, id of the instance in the same class/gloss.

* `signer_id`: *int*, id of the signer.

* `source`: *str*, a string identifier for the source site.

* `split`: *str*, indicates sample belongs to which subset.

* `url`: *str*, used for video downloading.

* `variation_id`: *int*, id for dialect (indexed from 0).

* `video_id`: *str*, a unique video identifier.

Please be kindly advised that if you decode with different FPS, you may need to recalculate the `frame_start` and `frame_end` to get correct video segments.

Constituting subsets
---------------
As described in the paper, four subsets WLASL100, WLASL300, WLASL1000 and WLASL2000 are constructed by taking the top-K (k=100, 300, 1000 and 2000) glosses from the `WLASL_vx.x.json` file.


Training and Testing

**Pose-TGCN**
```
cd WLASL
mkdir data
```
put all the videos under ```data/```.

Download [splits file](https://drive.google.com/file/d/16CWkbMLyEbdBkrxAPaxSXFP_aSxKzNN4/view?usp=sharing) and [body keypoints](https://drive.google.com/file/d/1k5mfrc2g4ZEzzNjW6CEVjLvNTZcmPanB/view?usp=sharing). Unzip them into ```SilentTalk/data```. You should see ```SilentTalk/data/splits``` and ```SilentTalk/data/pose_per_individual_videos``` folders.

To train the model, modify paths in ```train_tgcn.py main()``` to point to SilentTalk root.
```
python train_tgcn.py
```

To test the model, first download [pre-trained models](https://drive.google.com/file/d/1dzvocsaylRsjqaY4r_lyRihPZn0I6AA_/view?usp=sharing) and unzip to ```code/TGCN/archived```. Then run
```
python test_tgcn.py
```


Disclaimer
---------------
All the WLASL data is intended for academic and computational use only. No commercial usage is allowed. We highly respect copyright and privacy. If you find WLASL violates your rights, please contact us.




Citation
--------------

Please cite the WLASL paper if it helps your research:

     @inproceedings{li2020word,
        title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
        author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
        booktitle={The IEEE Winter Conference on Applications of Computer Vision},
        pages={1459--1469},
        year={2020}
     }

Please consider citing our work on WLASL.

    @inproceedings{li2020transferring,
     title={Transferring cross-domain knowledge for video sign language recognition},
     author={Li, Dongxu and Yu, Xin and Xu, Chenchen and Petersson, Lars and Li, Hongdong},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     pages={6205--6214},
     year={2020}
    }

Other works you might be interested in.

    @article{li2020tspnet,
      title={Tspnet: Hierarchical feature learning via temporal semantic pyramid for sign language translation},
      author={Li, Dongxu and Xu, Chenchen and Yu, Xin and Zhang, Kaihao and Swift, Benjamin and Suominen, Hanna and Li, Hongdong},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      pages={12034--12045},
      year={2020}
    }

    @inproceedings{li2022transcribing,
      title={Transcribing natural languages for the deaf via neural editing programs},
      author={Li, Dongxu and Xu, Chenchen and Liu, Liu and Zhong, Yiran and Wang, Rong and Petersson, Lars and Li, Hongdong},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={36},
      number={11},
      pages={11991--11999},
      year={2022}
    }

