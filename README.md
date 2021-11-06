# Title: Development of an algorithm for automatic detection of meniscus tears in knee magnetic resonance imaging (MRI) scans.

# Research aim
Utilising machine learning models as a computational technique for the diagnosis of osteoarthritis is still relatively new. The aim in this project is to develop a machine learning algorithm, and as proof of concept, to determine if the algorithm can identify meniscus tears by differentiating/ localizing abnormalities in MRI scans of the knee.

## Background
Osteoarthritis (OA) is the most prevalent medically treated arthritic condition worldwide. Diagnosis of symptomatic OA is usually made on the basis of clinical examination/ radiography and reported pain in the same joint. A meniscal tear is a frequent orthopaedic diagnosis and an early indication of OA. However, 61% of randomly selected subjects who showed meniscal tears in their knees during magnetic resonance imaging (MRI) scans, have not had any pain, aches, or stiffness during the previous months (Englund et al., 2008). So, meniscal tears are frequently encountered in both asymptomatic and symptomatic knees (Zanetti et al., 2003). Since an early detection of meniscal tear in asymptomatic knee appears to be an early indicator of OA and a risk factor for other articular structural changes, there is a need for a better and faster identification of meniscal tears (Ding et al., 2007). This is especially important in the absence of specialised radiologists, or a backlog due to increased use of medical imaging (McDonald et al., 2015, Kumamaru et al., 2018). 

## Data and Model
Data is from [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/). It consists of 1370 knee MRI exams performed at Stanford University Medical Center. The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports.

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




#### VGG16
![VGG16 architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/vgg16.png)



### Own models
Since the pretrained models were too complex for the dataset, I decided to build my own models instead. The models should be relatively smaller than the previous pretrained models.  

#### Adapted from AlexNet
![AlexNet architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/alexnet.png)



#### Model based on LeNet
![LeNet architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/ownmodel.png)




#### Functional API
![functional architecture](https://github.com/doscsy12/knee_mri_proj/blob/main/images/func_architecture.png)



