# AI-Powered Insider Threat Detection

## 📄 Abstract

This project presents an **AI-powered insider threat detection system** developed using the **CICIDS 2018 dataset**. It leverages **machine learning models**, specifically **Random Forest** and **Convolutional Neural Network (CNN)**, to classify network traffic as either benign or associated with various attack types. The solution includes:

- Comprehensive data preprocessing  
- Exploratory data analysis  
- Model training and evaluation  
- Visualization and interpretation  

The system achieved promising results in terms of accuracy and robustness, demonstrating the potential of AI in the domain of cybersecurity.

---

## 🔍 Introduction

Insider threats represent a critical concern for organizational cybersecurity, often resulting in severe data breaches and system compromises. These threats can be subtle, requiring advanced detection methods capable of understanding complex traffic behaviors.

The **CICIDS 2018 dataset**, a well-known benchmark in the intrusion detection research community, provides labeled network traffic, including multiple attack categories. This project utilizes the dataset to implement a machine learning-based detection system tailored to identify insider threats effectively.

---

## 📚 Literature Review

Several machine learning and deep learning approaches have been researched for intrusion detection:

- **Random Forest**: Robust, interpretable, and effective at feature ranking.  
- **Support Vector Machines (SVM)**: High accuracy but computationally intensive.  
- **Convolutional Neural Networks (CNNs)**: Good for capturing spatial/temporal patterns in data.

Key insights from previous work:
- Data preprocessing is essential: missing value handling, normalization, and class balancing.
- Deep learning models like CNNs, though originally for image tasks, are effective for sequential and structured data like network traffic.
- CICIDS 2018 remains a trusted benchmark for evaluating IDS solutions.

---

## ⚙️ Methodology

### 📊 Data Loading and Exploratory Data Analysis (EDA)

- Dataset loaded using `pandas`
- Analyzed:
  - Dataset shape & types
  - Missing values
  - Distribution of attack types
  - Feature correlations
- Visualizations:
  - Count plots
  - Correlation heatmaps
  - Scatter plots (interactive)

---

### 🧹 Data Preprocessing

1. Removed missing values
2. Encoded attack labels using `LabelEncoder`
3. Balanced dataset through class-wise resampling
4. Dropped non-numeric/identifier columns
5. Replaced infinite values with medians
6. Clipped outliers using IQR
7. Normalized features with `MinMaxScaler`
8. Performed stratified train-test split
9. Reshaped data for CNN (3D tensors)
10. Saved preprocessing artifacts for reuse

---

### 🧠 Model Development

#### ✅ Random Forest Classifier
- **Configuration**:
  - `n_estimators=100`, `max_depth=20`, adjusted `min_samples_split` and `min_samples_leaf`
- **Evaluation**:
  - Accuracy
  - Confusion matrix
  - Classification report
- **Feature Importance**:
  - Visualized top features
- **Deployment**:
  - Saved model for future inference

#### ✅ Convolutional Neural Network (CNN)
- **Architecture**:
  - 3 × `Conv1D` layers (64 → 128 → 256 filters)
  - Batch Normalization + MaxPooling after each
  - Dense layers with Dropout to prevent overfitting
- **Training**:
  - Loss: `categorical_crossentropy`
  - Optimizer: `Adam`
  - Epochs: 30 with EarlyStopping and ModelCheckpoint
- **Monitoring**:
  - Training/validation accuracy and loss plotted
- **Deployment**:
  - Saved best model

---

## ✅ Conclusion

The developed AI-based insider threat detection system effectively classifies and detects various attack types using the CICIDS 2018 dataset. Both the Random Forest and CNN models yielded high accuracy and generalizability, especially after rigorous preprocessing and feature engineering.

Key achievements:
- Effective handling of class imbalance
- High-performance ML and DL models
- Reproducibility through artifact storage
- Readiness for real-world deployment and monitoring tool integration

---

## 📚 References

1. **CICIDS 2018 Dataset** – Canadian Institute for Cybersecurity  
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32  
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep Learning*. Nature, 521(7553), 436–444  
4. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications  
5. [Scikit-learn Documentation](https://scikit-learn.org)  
6. [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)  

---

## 📎 Citation Link

[Project Source (Paste.txt)](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/62031108/faa0ee32-debb-44c8-b008-21e9ab2c4da1/paste.txt)
