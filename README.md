# Instance Segmentation Using Mask R-CNN

## Project Overview
This project focuses on instance segmentation using Mask R-CNN, applied to a custom dataset of poles and trashcans collected from a college campus. The goal is to accurately segment these objects using deep learning techniques.

## Dataset Collection and Annotation
We curated a custom dataset consisting of **200 images** captured from various locations within the college campus. The dataset primarily includes images of **poles and trashcans** under different lighting and background conditions.

For annotation, we utilized the **CVAT (Computer Vision Annotation Tool)** to label the dataset in **COCO format** using polygon annotations. This ensured precise boundary detection, enabling the model to learn intricate object structures effectively.

## Data Preprocessing
Before training the model, the dataset underwent several preprocessing steps:
- **Normalization:** Pixel values were normalized to improve convergence.
- **Data Augmentation:** Techniques such as flipping, rotation, and scaling were applied to increase model generalization.
- **Dataset Splitting:** The dataset was divided into training and validation sets to evaluate model performance.

## Model Architecture
We implemented **Mask R-CNN**, a state-of-the-art deep learning framework for instance segmentation. The model was built using:
- **ResNet-50 with Feature Pyramid Network (FPN) Backbone:** This backbone enhances feature extraction and improves detection accuracy across different scales.
- **Region Proposal Network (RPN):** To generate region proposals for detecting objects in images.
- **ROIAlign and Mask Head:** To refine segmentation masks with high precision.

## Training and Inference
The model was trained using the custom dataset with the following specifications:
- **Loss Function:** A combination of classification, bounding box regression, and mask loss.
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum.
- **Hyperparameters:** Tuned learning rate and weight decay for optimal performance.
- **Hardware:** Trained on GPU for accelerated computations.

After training, the model was tested on unseen images, and it successfully segmented poles and trashcans with high accuracy.

## Results and Observations
- The model demonstrated **robust segmentation capabilities** with minimal false positives.
- **Fine-tuning** on a pre-trained backbone significantly improved performance.
- Future improvements could include expanding the dataset and experimenting with different backbones like ResNeXt or EfficientNet.

## Conclusion
This project successfully demonstrates the application of **Mask R-CNN** for instance segmentation on a custom dataset. By leveraging **ResNet-50 with FPN**, the model effectively detects and segments poles and trashcans with high accuracy. Future work includes optimizing the model further and expanding its applicability to more object categories.


