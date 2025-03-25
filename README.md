# spice-classification
predicting spices in a provided input image


Methodology
This study employs a ResNet50-based deep learning model for spice image classification. The approach consists of data preprocessing, model training, inference, and optimization to enhance classification accuracy.

Dataset Preprocessing & Augmentation
ImageDataGenerator (TensorFlow) is used to apply transformations like rotation, shifting, shearing, zooming, and flipping to replicate real-world variations.

Normalization: Images are scaled to [0,1] for better model performance.

Validation & Test Sets: No additional scaling is applied.

Model Architecture & Fine-Tuning
Base Model: ResNet50 (pre-trained on ImageNet).

Fine-Tuning:

The last 140 layers are unfrozen for fine-tuning.

Earlier layers retain previously learned features.

Additional Layers:

GlobalAveragePooling2D

Dense (ReLU Activation)

Dropout (Regularization)

Softmax (Classification Layer)

Training Process
Epochs: 20

Batch Size: 32

Monitoring Performance:

Validation data tracks model performance.

Early stopping prevents overfitting.

Model Evaluation:

Assessed on test data for generalization capability.

Inference & Prediction
The trained model predicts spice categories for new images.

Preprocessing Steps:

Resizing images to 224×224

Normalization ([0,1] scaling)

Model Saving: Stored in HDF5 format for future inference without retraining.

Compilation & Optimization
Optimizer: Adam (learning rate = 0.00001) for gradual fine-tuning.

Loss Function: Categorical Cross-Entropy (for multi-class classification).

Performance Metric: Accuracy.

This transfer learning-based approach improves classification accuracy and enhances generalization to unseen spice images.

Dataset Link:https://data.mendeley.com/datasets/vg77y9rtjb/3
