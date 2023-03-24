# DORT

## Introduction

This is the offical implementation of paper [DORT: Modeling Dynamic Objects in Recurrent for Multi-Camera 3D Object Detection and Tracking]().

## Abstract

Recent multi-camera 3D object detectors usually leverage temporal information to construct multi-view stereo that alleviates the ill-posed depth estimation. However, they typically assume all the objects are static and directly aggregate features across frames. This work begins with a theoretical and empirical analysis to reveal that ignoring the motion of moving objects can result in serious localization bias. Therefore, we propose to model Dynamic Objects in RecurrenT (DORT) to tackle this problem. 
In contrast to previous global Bird-Eye-View (BEV) methods, DORT extracts object-wise local volumes for motion estimation that also alleviates the heavy computational burden. By iteratively refining the estimated object motion and location, the preceding features can be precisely aggregated to the current frame to mitigate the aforementioned adverse effects. The simple framework has two significant appealing properties. It is flexible and practical that can be plugged into most camera-based 3D object detectors. As there are predictions of object motion in the loop, it can easily track objects across frames according to their nearest center distances. Without bells and whistles, DORT outperforms all the previous methods on the nuScenes detection and tracking benchmarks with 62.5\% NDS and 57.6\% AMOTA, respectively.

<p align="center">
  <img src="figs/overview.png" height="200" />
</p>

## Main Results

We provide the main results on the nuScenes validation set with ResNet50 backbone.
<p align="center">

| config            | mAP      | NDS     | AMOTA     | AMOTP     |  
|:--------:|:----------:|:---------:|:--------:|:--------:|
| DORT-R50-704x256   | 37.9     | 52.1    | 42.4    | 1.264 |  
</p>




## Code Release
The code is still going through large refactoring. 

Please stay tuned for the clean release.

## Citation

```bibtex

@article{lian2023dort,
  title={DORT: Modeling Dynamic Objects in Recurrent for Multi-Camera 3D Object Detection and Tracking},
  author={Lian, Qing and Wang, Tai and Lin, Dahua and Pang, Jiangmiao},
  journal={arXiv preprint},
  year={2023}
}
```