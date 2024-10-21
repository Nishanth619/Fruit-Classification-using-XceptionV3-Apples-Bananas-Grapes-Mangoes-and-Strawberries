# Fruit Classification using XceptionV3

This project uses the **XceptionV3** model to classify images of five different types of fruits: **Apples, Bananas, Grapes, Mangoes, and Strawberries**. The model is trained using transfer learning and fine-tuned for improved accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to classify different types of fruits using deep learning. We utilize the XceptionV3 model, a pre-trained convolutional neural network, and fine-tune it on a custom dataset. This project demonstrates how transfer learning can be applied to image classification tasks for high accuracy.

## Dataset
The dataset used contains images of the following fruits:
1. Apples
2. Bananas
3. Grapes
4. Mangoes
5. Strawberries

The dataset is structured into subdirectories, where each folder contains images of a single fruit class. The data is split into **training** and **validation** sets.
Dataset Used:https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification
### Example Directory Structure:
\`\`\`


dataset/

    ├── train/
        ├── Apples/
        ├── Bananas/
        ├── Grapes/
        ├── Mangoes/
        └── Strawberries/

    └── validation/
        ├── Apples/
        ├── Bananas/
        ├── Grapes/
        ├── Mangoes/
        └── Strawberries/
\`\`\`

## Model Architecture
The XceptionV3 model is used as the base model, pre-trained on the ImageNet dataset. A Global Average Pooling layer and a Dense layer are added on top for classification. The model is fine-tuned by training only the added layers at first and optionally unfreezing some base model layers later for further fine-tuning.

### Key Layers:
- **Global Average Pooling**
- **Dense(1024 units, ReLU activation)**
- **Dense(5 units, Softmax activation)** (for 5 fruit classes)

## Installation
### 1. Clone the repository:
\`\`\`bash
git clone [https://github.com/Nishanth619/fruit-classification-xception.git](https://github.com/Nishanth619/Fruit-Classification-using-XceptionV3-Apples-Bananas-Grapes-Mangoes-and-Strawberries)
cd fruit-classification-xception
\`\`\`

### 2. Install required dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Prepare the dataset:
- Ensure that your dataset is organized into train and validation directories, as described above.

## Usage
1. **Training the model**: To start training the model, simply run:
   \`\`\`bash
   python train.py
   \`\`\`
   This will train the model on the dataset and save the trained model as \`fruit_classifier_xception.h5\`.

2. **Fine-tuning**: You can unfreeze layers of the Xception model to fine-tune for better accuracy. Modify the \`train.py\` script to enable fine-tuning.

3. **Model Evaluation**: After training, evaluate the model using the validation set.

## Results
- The model achieves high accuracy with transfer learning on the fruit classification dataset.
- Training and validation accuracy/loss plots are generated during training to monitor the model's performance.

### Example Results:
| Metric      | Value   |
|-------------|---------|
| Accuracy    | 90%+    |
| Loss        | Low     |

## Contributing
Contributions are welcome! If you have any suggestions, feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
" > README.md
