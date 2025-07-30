# 🔬 Skin-Disease and Skin Cancer - Classifier: AI-Powered Dermatological Diagnosis

This repository contains the research and implementation of a custom-built Convolutional Neural Network (CNN) designed for the accurate classification of 57 different types of skin diseases and cancers. Our model achieves a state-of-the-art accuracy of **96.64%**, offering a powerful tool to assist in dermatological diagnosis, aiming to surpass traditional methods in both speed and precision.

This project not only introduces a high-performance custom model but also provides a comprehensive comparative analysis against well-known pre-trained architectures like VGG16, MobileNet, and Inception V3.

## ✨ Core Features

-   **High-Precision Diagnosis:** Achieves **96.64% accuracy** in classifying 57 distinct classes of skin lesions.
-   **Custom CNN Architecture:** A bespoke model designed and optimized specifically for dermatological image data, outperforming standard pre-trained models.
-   **Comparative Analysis:** In-depth performance comparison against VGG16, MobileNet, Inception V3, and a standard Sequential CNN.
-   **Efficient & Fast:** Engineered for rapid image processing and classification, providing quicker results than traditional diagnostic approaches.
-   **Robust & Reliable:** Utilizes advanced data augmentation techniques and model tuning to ensure reliability and generalization in real-world scenarios.
-   **AI-Driven Healthcare:** Contributes to the advancement of AI in medicine, with a focus on improving early detection and treatment outcomes in dermatology.

## 🛠️ Tech Stack

| Category              | Technology / Library                                       |
| --------------------- | ---------------------------------------------------------- |
| **Language** | Python                                                     |
| **Deep Learning** | TensorFlow, Keras                                          |
| **Data Handling** | Pandas, NumPy                                              |
| **Data Augmentation** | `ImageDataGenerator` (from TensorFlow)                     |
| **Model Evaluation** | Scikit-learn (for metrics like confusion matrix, F1-score) |
| **Visualization** | Matplotlib, Seaborn                                        |

---

## ⚙️ Methodology & Workflow

Our approach is structured to build a robust and highly accurate classification model from the ground up.

1.  **Data Preparation & Augmentation:** To prevent overfitting and improve model generalization, we employ extensive data augmentation techniques using TensorFlow's `ImageDataGenerator`. This includes rotations, shifts, zooms, and flips to create a diverse and rich training dataset.
2.  **Custom CNN Model Architecture:** We designed a unique CNN architecture tailored for the specific features of skin lesion images. This custom model proved more efficient and accurate than generic, pre-trained models for this specific task.
3.  **Comparative Model Training:** We trained our custom model alongside several established architectures (VGG16, MobileNet, Inception V3, Sequential CNN) on the exact same dataset to create a fair and comprehensive performance benchmark.
4.  **Advanced Model Tuning:** The model was fine-tuned using techniques like learning rate scheduling, early stopping, and dropout regularization to achieve optimal performance.
5.  **Evaluation:** The model's final performance was evaluated based on key metrics, with a primary focus on classification accuracy.

## 📊 Performance & Results

Our custom-designed CNN model demonstrated superior performance across the board.

-   **Classification Accuracy:** **96.64%**
-   **Efficiency:** Our model processed images faster while maintaining higher precision compared to the benchmarked pre-trained models.

*(Placeholder for a confusion matrix or accuracy/loss graphs)*

```
[Insert a confusion matrix image or a plot showing training/validation accuracy curves here]
```

---

## 🚀 Getting Started

To replicate our findings or use the model for predictions, follow these steps.

### Prerequisites

-   Python 3.8 or newer
-   TensorFlow 2.x
-   An environment with GPU support is highly recommended for training.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/skin-lesion-classifier.git](https://github.com/your-username/skin-lesion-classifier.git)
    cd skin-lesion-classifier
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the Dataset:**
    *(Provide instructions or a link to download the dataset used for this research).*

### Running a Prediction

*(Provide a code snippet or command to run a prediction on a new image).*

```python
# Example prediction script
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('skin_lesion_classifier.h5')

# Load and preprocess an image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

print(f"Predicted Class Index: {predicted_class[0]}")
```

## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
