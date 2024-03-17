WLASL: A large-scale dataset for Word-Level American Sign Language (WACV 20' Best Paper Honourable Mention)
============================================================================================

This repository contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".

Please **star the repo** to help with the visibility if you find it useful.

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

To train the model, modify paths in ```train_tgcn.py main()``` to point to SilentTalk root.
```
python train_tgcn.py
```

To test the model, Then run
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

