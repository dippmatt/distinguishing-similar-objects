# Literature

## Approaches

The task is a supervised image classification problem of texture-less RGB data. Additionally, a depth channel is available in the T-LESS dataset, which could be used.

To improve the accuracy in object detection, three approaches are examined:

 - **Adapting the loss function** during model training: By implementing a loss function, that specifically penalizes wrong classifications of  similar object that belong to different classes, we hope to represent those objects further apart in latent space. We expect improved accuracy upon standard object detection loss-functions like cross-entropy.

 - **Adapting a deep learning model architecture**: Standard implementations of SOTA classification models like YOLO fittingly assume 3-channel (RGB) data as input. One possibility to improve classification accuracy is implementing a 4th channel which represents a binary mask of unique positions for similar objects. A simplified example of how such a mask could look like is shown below.

```
    Object 1, Class 1            Mask 1
     ┌─────────────┐         ┌─────────────┐
     │    ****     │         │             │  
     │  *      *   │         │             │  
     │ *        *  │         │             │  
     │ *        *  │         │             │  
     │ *        *  │         │             │  
     │ *        *  │         │             │  
     │  *      *   │         │             │  
     │    ****     │         │             │  
     └─────────────┘         └─────────────┘ 
                     
    Object 2, Class 2            Mask 2
     ┌─────────────┐         ┌─────────────┐
     │    ****     │         │             │  
     │  *      *   │         │             │  
     │ *        *  │         │             │  
     │ *        *  │         │             │  
     │ *        *  │         │             │  
     │ *  ****  *  │         │    1111     │  
     │  *      *   │         │             │  
     │    ****     │         │             │  
     └─────────────┘         └─────────────┘ 
      
```

- **Preprocessing**: We aim to investigate two approaches for preprocessing: Supervised and unsupervised approach.
    - **Unsupervised preprocessing**: The unsupervised approach does not need insights on the dataset used. Simple rules and filters for feature extraction should be applied to the input images before feeding it to the model to improve accuracy. The alterations to the input images should emphasize unique areas for similar objects. An example could be canny edge detection or harris corner detection. Furthermore, the detected corners could be used to create a mask, similar to the example above.
    - **Supervised preprocessing**: To only identify the unique areas in objects, knowledge about similar objects classes is needed. This means that the dataset and some kind of learning is needed for preprocessing. The example above shows why: To identify the unique crossbar in the ellipse of Object 2, the preprocessing algorithm needs to know that an object class (Object 1) without crossbar exists as well as a class with crossbar, to create Mask 2.

## Sota Research
### Loss Functions

**Contrastive Learning** is a technique that aims to identify and learn both similar and dissimilar features. The goal is exactly what we need: To drive the embeddings for different classes of similar objects further apart. Similar objects refers to objects that are close to each other in embedding space in an unsupervised scenario, regardless of their class.
Examples for loss functions for contrastive learning are: 
- Contrastive Loss [[1]](https://proceedings.mlr.press/r5/lecun05a/lecun05a.pdf).
- Triplet Loss

However, newer methods now represent state of the art:
- Supervised Contrastive Learning (2020) [[2]](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
    - Github: [SupContrast](https://github.com/HobbitLong/SupContrast)
- Tuned Contrastive Learning (2023) [[3]](https://arxiv.org/abs/2305.10675)
    - Github: no implementations yet, [Papers with code](https://paperswithcode.com/paper/tuned-contrastive-learning)


### Model Architectures

TODO

Github: 
 - [YOLOV8](https://github.com/ultralytics/ultralytics) (2023) - easy to use for developers!
 - [YOLOV9](https://github.com/WongKinYiu/yolov9) (2024)

### Preprocessing

TODO

Github:
- [Segment Anything](https://github.com/facebookresearch/segment-anything): Self Supervised object mask creation.
- [DINO](https://github.com/facebookresearch/dino): Self supervised vision transformer.
- [AutoAugment](https://github.com/DeepVoltaire/AutoAugment): Goal: Find best augmentation Policy for a given Dataset in a Search space of Policies and Sub-Policies.

## Own ideas and approaches

### Preprocessing

As the name implies, the T-LESS dataset does not have textures. Assuming 8-bit quantisation for each pixel of an R, G or B channel, an easy way to increase accuracy without modifying model architechture could be to misuse the 3 RGB channels of any RGB object classification model:
- First, place the grayscale version of the input in channel 1.
- To retain color information, reduce R, G and B color quantisation to 2 bit per channel, which sums up to 6 bit, leaving 2 unused bits for channel 2 of the model. These bits should not remain unused, but I don't know what to do with them for now.
- Use channel 3 to store the mask from [Approaches](#approaches) or any other preprocessing technique for the input.

Notes: 
1) Of course, the model will be trained using floating point input data, but by assuming 8-bit input data, a transition to embedded hardware and quantized models will be easier.
2) This approach requires complete retraining of the model used. No pretrained backbone will be helpful, since the input representation completely changed.