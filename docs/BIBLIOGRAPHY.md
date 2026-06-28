# Bibliography

This bibliography supports the academic framing of the project in semantic segmentation, autonomous-driving perception, convolutional encoder-decoder architectures, transformer-based segmentation, and class-imbalance handling.

## Semantic Segmentation and Urban Scene Understanding

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
2. Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE TPAMI.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
4. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV.
5. Yu, F., Koltun, V., & Funkhouser, T. (2017). Dilated Residual Networks. CVPR.

## Autonomous Driving Datasets and Road-Scene Perception

6. Brostow, G. J., Fauqueur, J., & Cipolla, R. (2009). Semantic Object Classes in Video: A High-Definition Ground Truth Database. Pattern Recognition Letters.
7. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., & Schiele, B. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. CVPR.
8. Geiger, A., Lenz, P., & Urtasun, R. (2012). Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR.
9. Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., & Beijbom, O. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR.

## Transformer-Based Segmentation

10. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS.
11. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention Mask Transformer for Universal Image Segmentation. CVPR.
12. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.

## Class Imbalance and Loss Functions

13. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. ICCV.
14. Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. (2017). Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. DLMIA.
15. Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. MLMI.

## Recommended Citation Style

When discussing this project academically, cite both the dataset and the model families used. For example, a concise report paragraph could state:

> The experimental analysis was conducted on CamVid, a road-scene semantic segmentation dataset, using CNN-based encoder-decoder architectures and transformer-based segmentation models. The evaluation focused on mIoU and class-wise IoU under a reduced safety-critical label space.
