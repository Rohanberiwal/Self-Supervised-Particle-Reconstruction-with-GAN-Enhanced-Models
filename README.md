# Self-Supervised Learning for End-to-End Particle Reconstruction Using Resnet15 and GAN model 

## Project Overview

This project involves training a ResNet-15 model for end-to-end particle reconstruction in the CMS (Compact Muon Solenoid) experiment using self-supervised learning. The project is divided into two main stages:

1. **Self-Supervised Learning**: Train a ResNet-15 model using the Barlow Twins self-supervised learning method on a provided unlabelled dataset.
2. **Finetuning and Evaluation**: Fine-tune the self-supervised model on a labeled dataset and compare its performance with a model trained from scratch.

The dataset generation includes synthetic data created using a GAN model fine-tuned on random images from PyTorch libraries. The final dataset consists of 3-channel images.

## Files Provided

- `self_supervised_training.ipynb`: Jupyter notebook containing the implementation for Barlow Twins training, finetuning, and evaluation.
- `model_weights/`: Directory containing the weights for both the self-supervised model and the fine-tuned model.
- `README.md`: This README file.

## Data Preparation

### Unlabelled Dataset (Pretraining Stage)
- Used for self-supervised learning.
- Generated using a GAN model fine-tuned on random images from PyTorch's image libraries.
- Consists of 3-channel images.

### Labelled Dataset (Finetuning Stage)
- Contains labeled images for supervised learning.
- The data split is 80% for training and 20% for evaluation.
- Ensure proper preprocessing and normalization.

## Self-Supervised Learning with Barlow Twins

Barlow Twins is a self-supervised learning method that focuses on maximizing the similarity between different augmented views of the same image while minimizing the redundancy between features.

**Steps**:
1. **Data Augmentation**:
   - Apply various augmentations to create different views of the same image.

2. **Model Architecture**:
   - Train a ResNet-15 model with Barlow Twins objectives, which include:
     - **Variance Minimization**: Ensuring features have a low variance.
     - **Invariance Maximization**: Ensuring similar views have similar representations.
     - **Covariance Regularization**: Reducing redundancy between feature dimensions.

3. **Training**:
   - Train the ResNet-15 model on the unlabelled dataset using Barlow Twins objectives.

4. **Save Model**:
   - Save the trained model weights for use in the finetuning stage.

## Finetuning and Evaluation

1. **Finetuning**:
   - Load the self-supervised ResNet-15 model and finetune it on the labeled dataset.
   - Training is performed with a low learning rate to adapt the model to the labeled data.

2. **Comparison**:
   - Train a ResNet-15 model from scratch on the labeled dataset for comparison.
   - Both models are evaluated on 20% of the data reserved for testing.

3. **Evaluation Metrics**:
   - Accuracy and loss are recorded for both models.
   - Ensure to avoid overfitting on the test dataset by using an independent sample.

## Jupyter Notebook Instructions

1. **Setup**:
   - Load the provided Jupyter notebook `self_supervised_training.ipynb`.
   - Ensure all dependencies (e.g., PyTorch, torchvision, Barlow Twins library) are installed.
   
2. **Run the Notebook**:
   - Follow the instructions in the notebook to run the code.
   - The notebook covers:
     - Training the ResNet-15 model with Barlow Twins on the unlabelled dataset.
     - Finetuning the model on the labeled dataset.
     - Comparing performance with a model trained from scratch.

3. **Model Weights**:
   - The `model_weights/` directory contains:
     - `self_supervised_model.pth`: Weights for the self-supervised ResNet-15 model.
     - `finetuned_model.pth`: Weights for the finetuned ResNet-15 model.

## Notes

- Ensure to handle data preprocessing, augmentation, and normalization appropriately.
- Monitor for overfitting and use an independent sample for validation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

