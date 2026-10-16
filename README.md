# Impact of Data Scale on the Performance of Quantum Feature Maps for Breast Cancer Ultrasound Classification

## Introduction

Breast cancer is the most commonly diagnosed malignancy among women, and early diagnosis is a determining factor for a favorable prognosis. Convolutional Neural Networks (CNNs) have been widely employed in diagnostic image processing to support the classification of clinical findings and to enable more accurate diagnoses [1]. The LeNet architecture, in particular, has shown a strong ability to extract discriminative features with high accuracy, thereby contributing to the detection and early diagnosis of breast cancer. Despite the advances these models have brought to diagnostic support, limitations persist that are associated with the heterogeneity, irregularity, and high dimensionality of the data. In this context, Quantum Machine Learning (QML) emerges as a promising approach [2], especially when employed in hybrid quantum-classical architectures. QML aims to integrate the quantum properties of superposition and entanglement into the classification of data whose complexity exceeds the processing capacity of conventional classical methods.

This study is motivated by the need for a comparative evaluation between Quantum Transfer Learning (QTL) models and the classical CNN approach across different data volumes. To this end, three breast ultrasound image datasets of different sizes will be used to classify breast lesions as benign or malignant.

## Methodology

The methodology comprises the set of steps adopted for the development and evaluation of the Quantum Transfer Learning model applied to the classification of breast cancer ultrasound images.

1. **Data Preparation and Preprocessing:** The study will use data at three different scales, consisting of breast ultrasound images [3]–[5]. A data-cleaning stage will be carried out to ensure the quality and integrity of the training set. This stage includes removing duplicate or corrupted images, excluding samples with inconsistent or missing annotations, and checking for visual artifacts that could introduce noise into model learning.

2. **Classical Baseline Model:** The baseline model will be implemented with the LeNet architecture and trained on the largest dataset, in order to maximize its ability to extract relevant features from the ultrasound images. After training, the remaining datasets will be fed to the model for performance evaluation, making it possible to analyze how the architecture behaves across different data volumes.

3. **Feature Map Extraction and Application:** Feature maps from the convolutional layers of the CNN will be extracted and used as encoded representations of the input data. This step investigates which feature map configurations yield representations best suited to quantum circuits, evaluating their effectiveness in terms of accuracy and performance relative to the classical reference model.

4. **QNN Implementation:** A Quantum Neural Network (QNN) will be developed in which the feature maps obtained in the previous step serve as data-encoding parameters for the quantum circuit, using the Quantum Transfer Learning technique. The model will be structured to allow a systematic evaluation of the feasibility and performance of the hybrid approach.

5. **Quantum Transfer Learning Architecture:** This step reuses the LeNet CNN as a feature extractor. To that end, the weights of the convolutional layers will be frozen, preserving the representations learned in the previous phase. A bottleneck layer, consisting of a classical linear layer, is inserted after the feature extractor to reduce the LeNet output vector to a dimension compatible with the number of available qubits. The classical values produced by the bottleneck layer are then mapped into Hilbert space through quantum feature maps. Following the methodology adopted in [6], multiple encoding schemes will be evaluated and optimized.
   To identify the encoding scheme best suited to the problem, a systematic experimentation pipeline will be implemented: a single parameterizable QNN script will accept different feature maps as input and record the performance metrics for each configuration. The results will be stored in a structured format, enabling direct comparison between encoding schemes and the identification of the one that generalizes best across the evaluated datasets.

6. **Validation and Local Simulation:** Before execution on real quantum hardware, the model will be validated locally through classical simulation, a strategy that helps detect architectural inconsistencies and select the most promising configurations. This stage will be implemented with the Qiskit library [7]. The simulation will run in noisy mode, incorporating noise models to approximate real quantum hardware conditions as closely as possible and thus ensure that the resulting metrics are representative of the performance expected on physical execution. At the end of this phase, the five best-performing feature maps in simulation will be selected for execution on real quantum hardware, as described in the next step.

7. **Quantum Hardware Execution:** The quantum circuits developed will be deployed and executed on a real QPU made available by IBM through its cloud quantum computing platform (IBM Quantum) [8].

8. **Comparative Analysis and Final Report:** The results obtained throughout the previous steps will be consolidated and critically analyzed to determine whether, across different dataset sizes, the hybrid QTL approach with quantum feature maps offers advantages or disadvantages relative to the classical CNN model. The findings will be summarized in a final report discussing the feasibility, limitations, and future perspectives of the quantum approach for image classification tasks.

## Completed Steps

| # | Step | Status |
|---|------|:------:|
| 1 | Data Preparation and Preprocessing | Completed |
| 2 | Classical Baseline Model (LeNet) |  Completed |
| 3 | Feature Map Extraction and Application |  Completed |
| 4 | QNN Implementation |  Pending |
| 5 | Quantum Transfer Learning Architecture |  Pending |
| 6 | Validation and Local Simulation |  Pending |
| 7 | Quantum Hardware Execution |  Pending |
| 8 | Comparative Analysis and Final Report |  Pending |

## Data Usage and Acknowledgements

This work uses the following datasets, and the authors gratefully acknowledge their original creators:

- **BUSI:** W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy [9].
- **BUS-BRA:** W. Gómez-Flores, M. J. Gregorio-Calas, and W. C. de Albuquerque Pereira [10].
- **GDPH&SYSUCC:** Y. Mo *et al.* [11].

## References

[1] S. Balasubramaniam *et al.*, "A modified LeNet CNN for breast cancer diagnosis in ultrasound images," Dept. Comput. Sci. Eng., Sona College of Technology, Salem, India, 2023.

[2] D. Martín Pérez *et al.*, "Hybrid classical-quantum transfer learning with noisy quantum circuits," Data Sci. Big Data Lab, Pablo de Olavide Univ., Seville, Spain, 2026.

[3] V. A. Sairam, "Ultrasound breast images for breast cancer," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer

[4] "Breast ultrasound images dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

[5] "A curated benchmark dataset for ultrasound based breast lesion analysis," The Cancer Imaging Archive. [Online]. Available: https://www.cancerimagingarchive.net/collection/breast-lesions-usg/

[6] S. M. Abdullah *et al.*, "Hybrid quantum–classical k-means clustering via quantum feature maps," Dept. Comput. Sci., Lahore Univ. Manage. Sci., Lahore, Pakistan, 2026.

[7] A. Javadi-Abhari *et al.*, "Quantum computing with Qiskit," 2024, *arXiv:2405.08810*. [Online]. Available: https://arxiv.org/abs/2405.08810

[8] S. Seelam *et al.*, "Reference architecture of a quantum-centric supercomputer," IBM T. J. Watson Research Center, Yorktown Heights, NY, USA, 2026.

[9] W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, "Dataset of breast ultrasound images," *Data in Brief*, vol. 28, Art. no. 104863, 2020, doi: 10.1016/j.dib.2019.104863.

[10] W. Gómez-Flores, M. J. Gregorio-Calas, and W. C. de Albuquerque Pereira, "BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems," *Med. Phys.*, 2024, doi: 10.1002/mp.16812.

[11] Y. Mo *et al.*, "HoVer-Trans: Anatomy-aware HoVer-Transformer for ROI-free breast cancer diagnosis in ultrasound images," 2022.

[12] V. Bergholm *et al.*, "PennyLane: Automatic differentiation of hybrid quantum-classical computations," 2018, *arXiv:1811.04968*. [Online]. Available: https://arxiv.org/abs/1811.04968