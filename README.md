# Attack-Bard




## Introduction

---

Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. 
By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., 26% attack success rate against Bing Chat and 86\% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. 


We provide codes for 4 experiments.

1. attack_img_encoder_misdescription.py: Image embedding attack against Bard's image description.

2. attack_vlm_misclassify.py: Text description attack against Bard's image description.

3. attack_img_encoder_nsfw.py: Attack on toxic detection.

4. attack_bomb.py: This one is not included in our paper. This is to let the MLLM output "bomb bomb...".


## Getting Started

---

### Installation

- Configurate the environment, vicuna weights, following the instruction in https://github.com/Vision-CAIR/MiniGPT-4    
- Prepare NFSW dataset. Put your NFSW images into "./resources/NSFW/ger_porn" folder (Refer to attack_img_encoder_nsfw.py for detail). 
- Prepare NIPS17 dataset. Download NIPS17 dataset from "https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set". Unzip it into "./resources/NIPS17". (Refer to ./data/NIPS17.py for more detail)


### Run the code

Run:
```
CUDA_VISIBLE_DEVICES=0 attack_bomb.py
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2 attack_img_encoder_misdescription.py
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2 attack_img_encoder_nsfw.py
```
or
```
CUDA_VISIBLE_DEVICES=0 attack_vlm_misclassify.py
```


### Results

- Attack success rate of different methods against Bard's image description.


|                         | Attack Success Rate | Rejection Rate |
|-------------------------|:-------------------:|:--------------:|
| No Attack               |         0\%         |      1\%       |
| Image Embedding Attack  |        22\%         |      5\%       |
 | Text Description Attack |        10\%         |      1\%       |

- We achieve 36\% attack success rate against Bard's toxic detector.




# Acknowledgement

---

Our code is implemented based on [**MiniGPT4**](https://github.com/Vision-CAIR/MiniGPT-4) and [**AdversarialAttacks**](https://github.com/huanranchen/AdversarialAttacks).  Thanks them for supporting! 



