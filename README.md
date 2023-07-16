# Temporal Enhanced Training of Multi-view 3D Object Detector via Historical Object Prediction

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-enhanced-training-of-multi-view-3d/3d-object-detection-on-nuscenes-camera-only)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-camera-only?p=temporal-enhanced-training-of-multi-view-3d)

This repo is the official implementation of ["Temporal Enhanced Training of Multi-view 3D Object Detector via Historical Object Prediction"](https://arxiv.org/abs/2304.00967) by Zhuofan Zong, Dongzhi Jiang, Guanglu Song, Zeyue Xue, Jingyong Su, Hongsheng Li, and Yu Liu.


## News

* ***[07/14/2023]*** HoP is accepted to ICCV 2023!
* ***[04/05/2023]*** HoP achieves new SOTA performance on [nuScenes 3D detection leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera) with **68.5 NDS** and **62.4 mAP**.
   

## Introduction

In this paper, we propose a new paradigm, named Historical Object Prediction (HoP) for multi-view 3D detection to leverage temporal information more effectively. The HoP approach is straightforward: given the current timestamp t, we generate a pseudo Bird's-Eye View (BEV) feature of timestamp t-k from its adjacent frames and utilize this feature to predict the object set at timestamp t-k. Our approach is motivated by the observation that enforcing the detector to capture both the spatial location and temporal motion of objects occurring at historical timestamps can lead to more accurate BEV feature learning. First, we elaborately design short-term and long-term temporal decoders, which can generate the pseudo BEV feature for timestamp t-k without the involvement of its corresponding camera images. Second, an additional object decoder is flexibly attached to predict the object targets using the generated pseudo BEV feature. Note that we only perform HoP during training, thus the proposed method does not introduce extra overheads during inference. As a plug-and-play approach, HoP can be easily incorporated into state-of-the-art BEV detection frameworks, including BEVFormer and BEVDet series. Furthermore, the auxiliary HoP approach is complementary to prevalent temporal modeling methods, leading to significant performance gains. Extensive experiments are conducted to evaluate the effectiveness of the proposed HoP on the nuScenes dataset. We choose the representative methods, including BEVFormer and BEVDet4D-Depth to evaluate our method. Surprisingly, HoP achieves 68.5% NDS and 62.4% mAP with ViT-L on nuScenes test, outperforming all the 3D object detectors on the leaderboard.


## Cite HoP

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{hop2023,
      title={Temporal Enhanced Training of Multi-view 3D Object Detector via Historical Object Prediction},
      author={Zhuofan Zong and Dongzhi Jiang and Guanglu Song and Zeyue Xue and Jingyong Su and Hongsheng Li and Yu Liu},
      year={2023},
      eprint={2304.00967},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.