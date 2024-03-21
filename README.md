# image_segmentation_optimization

## Goal
- Gain the ability to select the best optimizer for deep learning-based medical image segmantation
- Understand what the optimization is in terms of deep learning
### Stretch Goal
- Implement more accurate and faster optimizer

## Background
### Segmentation architectures
#### [U-Net architecture](#ref-4)
- Known for precise segmentation with smaller size of traning dataset
### [Optimizer](#ref-4)
1. SGD
    - Optimizer with fixed LR/MR
2. AdaGrad
    - Optimizer with adaptive LR
3. ADAM
    - Optimizer with adaptive LR/MR
### [Evaluation](#ref-3)
- Dice Similarity Coefficient (DSC)
    - F-meature based metrics
- Cohen's Kappa (Kap)
- Average Hausdorff Distance (AHD)
### Medical Image
#### Heart
- Small traning dataset with large variability
#### Chromosome
- Large traning dataset
- Main challenge is the separation of overlapping chromosomes

## Success Measure
- Draw a comparison table for each optimizer with its evaluation metrics

## Timeline
| Date          | Task          |
| ------------- | ------------- |
| Week1 (1/15)  | Look for proper dataset for medical image segmentation |
| Week2 (1/22)  | Implement Deep Learning model for the chosen dataset and learn its syntax |
| Week3 (1/29)  | Continue implementing Deep Learning model for the chosen dataset |
| Week4 (2/5)  | Modularize jupyter notebook / Setup environment for server |
| Week5 (2/12)  | Modularize a monolithic code base |
| Week6 (2/19)  | Implement evaluation methods |
| Week7 (2/26)  | Test model with a larger data in the server & Setup environment for server |
| Week8 (3/4)   | Midterm |
| Week9 (3/11)  | Spring break |
| Week9 (3/18)  | Implement SGD optimizer |
| Week10 (3/25)  | Implement AdaGrad optimizer |
| Week11 (4/1)  | Implement ADAM optimizer |

## Question

## TODO List
- Ensure setup environment for KU server

## Reference
### Research Paper
1. <a href="https://arxiv.org/pdf/2209.05414v1.pdf" id="ref-1">Chromosome Segmentation Analysis Using Image Processing Techniques and Autoencoders</a>
1. <a href="https://link.springer.com/article/10.1007/s10278-019-00227-x" id="ref-2">Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges</a>
1. <a href="https://bmcresnotes.biomedcentral.com/articles/10.1186/s13104-022-06096-y" id="ref-3">Towards a guideline for evaluation metrics in medical image segmentation</a>
1. <a href="https://www.frontiersin.org/articles/10.3389/fradi.2023.1175473/full" id="ref-4">Selecting the best optimizers for deep learning–based medical image segmentation</a>
1. <a href="https://link.springer.com/article/10.1007/s10278-019-00227-x" id="ref-5">Deep Learning Techniques for Medical Image Segmentation: Achievements and Challenges</a>
1. [A survey of deep learning optimizers - first and second order methods](https://arxiv.org/pdf/2211.15596.pdf)
1. [A large annotated medical image dataset for the development and evaluation of segmentation algorithms](https://arxiv.org/pdf/1902.09063v1.pdf)
1. [An Open Dataset of Annotated Metaphase Cell Images for Chromosome Identification](https://www.nature.com/articles/s41597-023-02003-7#Sec6)


### Dataset
- [Medical Segmentation Decathlon Dataset](http://medicaldecathlon.com)
- [MoNuSeg Data](https://monuseg.grand-challenge.org/Data/)


### Code
#### Image Segmentation
- [MedicalSegmentationDecathlon](https://github.com/Soft953/MedicalSegmentationDecathlon)
- [Dataloader for semantic segmentation](https://discuss.pytorch.org/t/dataloader-for-semantic-segmentation/48290)
#### Optimization
- [Stochastic variance reduced algorithms : Implementation of SVRG and SAGA optimization algorithms for deep learning](https://github.com/kilianFatras/variance_reduced_neural_networks)
- [Custom Optimizer in PyTorch](https://discuss.pytorch.org/t/custom-optimizer-in-pytorch/22397)
- [PyTorch: OPTIMIZING MODEL PARAMETERS](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch: TORCH.OPTIM](https://pytorch.org/docs/stable/optim.html)