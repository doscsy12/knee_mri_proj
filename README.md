# Title: Development of an algorithm for automatic detection of meniscus tears in radiographic images of the knee

# Problem Statement
Utilising machine learning models as a computational technique for the diagnosis of osteoarthritis is still relatively new. The aim in this project is to develop a machine learning algorithm, and to determine if the algorithm can identify meniscus tears by differentiating/ localizing abnormalities in radiographic images of the knee.

## Background
Osteoarthritis (OA) is the most prevalent medically treated arthritic condition worldwide. Diagnosis of symptomatic OA is usually made on the basis of clinical examination/ radiography and reported pain in the same joint. A meniscal tear is a frequent orthopaedic diagnosis and an early indication of OA. However, 61% of randomly selected subjects who showed meniscal tears in their knees during magnetic resonance imaging (MRI) scans, have not had any pain, aches, or stiffness during the previous months (Englund et al., 2008). So, meniscal tears are frequently encountered in both asymptomatic and symptomatic knees (Zanetti et al., 2003). Since an early detection of meniscal tear in asymptomatic knee appears to be an early indicator of OA and a risk factor for other articular structural changes, there is a need for a better and faster identification of meniscal tears (Ding et al., 2007), so that treatment options are made available early.

## Data and Model
Data is from [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/). It consists of 1370 knee MRI exams performed at Stanford University Medical Center. The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports.

## Notebooks
| modeling_notebook      | model | dataset   | diagnosis |
|------------------------|-------|-----------|-----------|
| knee_ownmodel          | own   | extracted | all       |
| meniscus_ownmodel      | own   | extracted | meniscus  |
| meniscus_own_full_data | own   | full      | meniscus  |
| meniscus_vgg16         | vgg16 | extracted | mensicus  |

