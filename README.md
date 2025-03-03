# Diagnosis of Skin Cancer Using Machine Learning

## 📌 Overview
Skin cancer is one of the most common cancers worldwide. Early detection is crucial for effective treatment. This project leverages **machine learning (ML) and deep learning (DL) techniques** to classify skin lesions into different categories based on dermoscopic images.

## 🚀 Features
- **Automated Classification**: Uses ML and DL models to distinguish between benign and malignant skin lesions.
- **Pretrained Models**: Implements Transfer Learning with models like **ResNet-50, EfficientNet, and InceptionV3**.
- **Web App Deployment**: Uses **Flask/FastAPI** for real-time predictions.
- **Data Augmentation**: Improves model generalization by applying image transformations.
- **Explainability**: Uses **Grad-CAM/SHAP** to visualize important features.

---

## 📂 Dataset
The dataset used for this project is taken from Kaggle:
- **Skin Cancer: Malignant vs. Benign Dataset** from Kaggle: [Dataset Link](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)

### 🔹 Data Preprocessing
- **Image Resizing**: Standardizing to **224×224 pixels**
- **Data Augmentation**: Flipping, rotation, contrast enhancement, etc.
- **Normalization**: Scaling pixel values to [0,1]
- **Class Balancing**: Using **SMOTE/Oversampling** for handling data imbalance

---

## 🏗️ Machine Learning Pipeline
### **Step 1: Feature Extraction**
✅ **Traditional Methods**
- Color and texture features using **GLCM, Fourier Descriptors**

✅ **Deep Learning-Based**
- Feature extraction from **CNN models**

### **Step 2: Model Selection & Training**
✅ **Machine Learning Models**
- Support Vector Machine (**SVM**)
- Logistic Regression
- Decision Tree

✅ **Deep Learning Models**
- **CNN (Custom Architecture)**
- **Pretrained Models:** ResNet-50, InceptionV3, EfficientNet
- **Hybrid Model:** CNN + SVM

### **Step 3: Model Evaluation**
Metrics used:
- **Accuracy**
- **Precision, Recall, and F1-Score**
- **ROC-AUC Curve**
- **Confusion Matrix**

---

## 🎯 Deployment
The trained model is deployed using:
- **Flask/FastAPI** for building a web API.
- **Streamlit** for interactive UI.
- **TensorFlow Lite/ONNX** for mobile deployment.

---

## 📌 Installation & Usage
### **🔹 Prerequisites**
Ensure you have Python 3.8+ installed. Required libraries:
```bash
pip install numpy pandas matplotlib tensorflow keras scikit-learn opencv-python flask streamlit
```

### **🔹 Running the Project**
1. Clone the repository:
```bash
git clone https://github.com/yourusername/skin-cancer-diagnosis.git
cd skin-cancer-diagnosis
```
2. Run the training script:
```bash
python train.py
```
3. Start the web application:
```bash
python app.py
```
4. Open the browser and go to:
```
http://127.0.0.1:5000
```

---

## 🛠️ Challenges & Future Work
### **Challenges:**
- **Data Imbalance:** Some skin cancer types are rarer than others.
- **Similar Lesions:** Certain benign and malignant lesions look similar.
- **Explainability:** Improving AI model interpretability for better trust.

### **Future Enhancements:**
✅ **Incorporate 3D Imaging for better analysis**
✅ **Deploy on mobile (Edge AI using Jetson Nano, Raspberry Pi)**
✅ **Multimodal learning (combining patient history + images)**

---

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. Your contributions are welcome!

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📬 Contact
For any queries, feel free to reach out:
📧 Email: athawalediksha2002@gmail.com
🔗 LinkedIn: [Diksha Athawale](https://www.linkedin.com/in/dikshaathawale/)
