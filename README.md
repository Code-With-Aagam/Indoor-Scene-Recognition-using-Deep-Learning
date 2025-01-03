# Indoor Scene Recognition using Deep Learning

This project aims to classify indoor scenes using deep learning techniques. It leverages a dataset of 67 indoor categories with over 15,000 images and employs a ResNet18 model for image classification.

## Features

* Classifies indoor scenes into 67 categories.
* Utilizes transfer learning with a pre-trained ResNet18 model.
* Achieves an accuracy of up to 76% on the test dataset.
* Includes functionality for predicting external images.

## Requirements

* Python 3.7+
* PyTorch 1.10+
* torch-vision 0.11+
* CUDA 9.2+ (optional for GPU acceleration)
* Other dependencies listed in `requirements.txt`

To install the necessary libraries, run:
- bash pip install -r requirements.txt
- **requirements.txt**
* torch==1.10.2+cu113  # Or the specific PyTorch version you used
* torchvision==0.11.3+cu113  # Or the specific torch-vision version you used
* matplotlib   # For plotting
* Numpy   # For numerical operations
* Pillow   # For image processing

## Methodology

The project follows these steps:

1. **Data Loading and Preprocessing:** The MIT Indoor Scenes dataset is loaded and preprocessed using torchvision transforms, including resizing and converting to tensors.
2. **Data Splitting:** The dataset is split into training, validation, and test sets to ensure proper model evaluation.
3. **Model Selection:** A pre-trained ResNet18 model is chosen as the base architecture due to its proven performance in image classification tasks.
4. **Transfer Learning:** The ResNet18 model is fine-tuned on the indoor scene recognition dataset using transfer learning techniques. The final fully connected layer is modified to output predictions for the 67 categories.
5. **Model Training:** The model is trained using the Adam optimizer and cross-entropy loss function. Training progress is monitored using metrics like accuracy and loss.
6. **Model Evaluation:** The trained model is evaluated on the validation and test sets to assess its performance and generalization capabilities.
7. **Prediction:** The model can be used to predict the category of new indoor scene images using the `predict.py` script.

## Dataset Usage

The project utilizes the MIT Indoor Scenes dataset, which can be found [here](https://drive.google.com/drive/folders/1c6EGMntT1gmkJSem52zoNhO21zU8XrWO). This dataset consists of:

* **67 Indoor Categories:** Including airport_inside, artstudio, bakery, barbershop, and more.
* **15,620 Images:** With at least 100 images per category.
* **Image Format:** All images are in JPG format.

The dataset is split into training, validation, and test sets with the following proportions:

* **Training Set:** 13,000 images (83.2%)
* **Validation Set:** 2,000 images (12.8%)
* **Test Set:** 620 images (4.0%)

## Results

The trained ResNet18 model achieved the following results:

* **Validation Accuracy:** Up to 76%
* **Test Accuracy:** Comparable to validation accuracy, demonstrating good generalization.

The model's predictions on external images were also qualitatively evaluated and found to be satisfactory.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
