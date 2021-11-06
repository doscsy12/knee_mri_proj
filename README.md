# Title: Development of an algorithm for automatic detection of meniscus tears in knee magnetic resonance imaging (MRI) scans.

# Research aim
Utilising machine learning models as a computational technique for the diagnosis of osteoarthritis is still relatively new. The aim in this project is to develop a machine learning algorithm, and as proof of concept, to determine if the algorithm can identify meniscus tears by differentiating/ localizing abnormalities in MRI scans of the knee.

## Background
Osteoarthritis (OA) is the most prevalent medically treated arthritic condition worldwide. Diagnosis of symptomatic OA is usually made on the basis of clinical examination/ radiography and reported pain in the same joint. A meniscal tear is a frequent orthopaedic diagnosis and an early indication of OA. However, 61% of randomly selected subjects who showed meniscal tears in their knees during magnetic resonance imaging (MRI) scans, have not had any pain, aches, or stiffness during the previous months (Englund et al., 2008). So, meniscal tears are frequently encountered in both asymptomatic and symptomatic knees (Zanetti et al., 2003). Since an early detection of meniscal tear in asymptomatic knee appears to be an early indicator of OA and a risk factor for other articular structural changes, there is a need for a better and faster identification of meniscal tears (Ding et al., 2007). This is especially important in the absence of specialised radiologists, or a backlog due to increased use of medical imaging (McDonald et al., 2015, Kumamaru et al., 2018). 

## Data and Model
Data is from [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/). It consists of 1370 knee MRI exams performed at Stanford University Medical Center. The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports. The original dataset is 5D tensor with shape (batch_shape, *s*, conv_dim1, conv_dim2, channel), where *s* is the number of slices per scan. The training set consists of 1130 MRI images from coronal, sagittal and axial planes. The validation set consists of 120 images from coronal, sagittal and axial planes. 

## Notebooks
| notebook            | model          | dataset         | diagnosis               |
|---------------------|----------------|-----------------|-------------------------|
| meniscus_resnet50   | resnet50       | extracted three | meniscus                |
| meniscus_vgg16      | vgg16          | extracted three | meniscus                |
| meniscus_alexnet    | alexnet        | extracted three | meniscus                |
| meniscus_ownmodel   | own            | extracted one   | meniscus                |
| kneeone_ownmodel    | own            | extracted one   | meniscus, acl, abnormal |
| meniscus_fulldata   | own            | fulldata        | meniscus                |
| meniscus_functional | own functional | extracted one   | meniscus                |

## Models
### Transfer Learning
Transfer learning is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could be applied when trying to recognize trucks ([source](https://en.wikipedia.org/wiki/Transfer_learning)).

In this project, I will discuss Resnet50 & VGG16. Keras offers many other [models](https://keras.io/api/applications/). 

#### ResNet50
![ResNet50 architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/resnet.png)
ResNet-50 is a convolutional neural network that is 50 layers deep. It was first introduced in 2015 ([He et al., 2015](https://arxiv.org/abs/1512.03385)), and become really popular after winning several classification competitions. Since it is FIFTY layers deep, the stacked layers can enrich the features of the models. However, it can also lead to degradation (ie, the accuracy layers may saturate and then slowly degrade after a point). Thus, model performance might deteriorate on both training and testing sets. 
<br>
<br> Unfortunately, when using a pretrained model, one is limited by input_shape. For ResNet50, it should be in 4D tensor with shape (batch_shape, conv_dim1, conv_dim2, channel). Thus, middle three images were extracted from each MRI scan, and used as inputs into the model. Only the last convolutional block of ResNet and added fully connected layers were trained. Other layers were frozen. Due to the complexity of the model, there was a tendency to overfit. Tuning of kernel_regularizer and dropout parameters were used to minimise overfitting. EarlyStopping and ModelCheckpoint were used to monitor validation loss, and training was stopped if there was no changes to validation loss. Despite best attempts, the average validation accuracy is 0.567. Due to the deep layers, this model was considered too complex for the dataset. 

#### VGG16
![VGG16 architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/vgg16.png)
The VGG model was created by [Oxford Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) (VGG), which helped fueled transfer learning work on new predictive modeling tasks.

Despite winning the ImageNet challenge in 2014, the VGG models (VGG16, VGG19) are no longer considered state-of-the-art. However, they are still powerful pre-trained models that are useful as the basis to build better predictive image classifiers, and a good foundation for this project.
<br>
<br> Similarly to ResNet, the input shape for VGG16 should be in 4D tensor with shape (batch_shape, conv_dim1, conv_dim2, channel). Thus, middle three images were extracted from each MRI scan, and used as inputs into the model. Only the last convolutional block of VGG16 and added fully connected layers were trained. Other layers were frozen. Despite having only 23 layers, there was a tendency to overfit. Kernel_regularizer, batch normalisation and dropout were tuned to minimise overfitting. However, a GlobalAveragePooling layer with a Dropout of 0.6 worked well. However, average accuracy remains poor, at approximately 0.56-0.58 for all three planes. Due to the overfitting, VGG16 was also considered too complex for the dataset.

### Own models
Since the pretrained models were too complex for the dataset, I decided to build my own models instead. The models should be relatively smaller than the previous pretrained models.  

#### Adapted from AlexNet
![AlexNet architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/alexnet.png)



#### Model based on LeNet
![LeNet architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/ownmodel.png)




#### Functional API
![functional architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/func_architecture.png)



