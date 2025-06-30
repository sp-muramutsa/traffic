# 🚦 Traffic Sign Classification: Deep Learning with GTSRB

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![NumPy](https://img.shields.io/badge/numpy-1.26.4-blueviolet)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-red)
![pandas](https://img.shields.io/badge/pandas-2.2.2-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-yellowgreen)

![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## 📖 Project Overview

This project builds and compares **deep learning models** for classifying traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The task is a multi-class classification problem with 43 traffic sign categories.

The goal is to explore how different model architectures — from basic MLPs to pretrained CNNs — perform on real-world traffic images. This has key applications in:

* Autonomous vehicle perception
* Real-time road sign recognition
* Embedded systems for road safety

---

## 🗃️ Dataset

* **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)
* **Size:** Thousands of labeled images in 43 categories
* **Properties:**

  * Varying shapes, lighting, and angles
  * Images resized to `30x30`, `96x96`, or `224x224` depending on model
* **Target:** Traffic sign class (integer from 0 to 42)

---

## 🔧 Data Preprocessing & Feature Engineering

* Resized images to model-compatible shapes
* Normalized image pixel values (`[0, 1]`) or used model-specific preprocessing (`preprocess_input`)
* Converted label folders into numerical class labels
* Created efficient TensorFlow datasets with batching and prefetching for GPU optimization
* Visualized sample images with `matplotlib` and verified image shapes

---

## ⚙️ Modeling & Evaluation

| Model                      | Train Accuracy | Test Accuracy | Notes                                                      |
| -------------------------- | -------------- | ------------- | ---------------------------------------------------------- |
| **Model 0:** MLP           | \~91%          | \~90%         | Baseline dense model. Fast but limited spatial awareness.  |
| **Model 1:** CNN           | \~99%          | \~99%         | Lightweight CNN with pooling. Strong generalization.       |
| **Model 2:** CNN + Dropout | \~99%          | \~99%         | Regularized CNN. Similar accuracy, slower training.        |
| **Model 3:** ResNet50V2    | \~2%           | \~2–3%        | Underfit due to frozen layers or misaligned preprocessing. |
| **Model 4:** MobileNetV2   | \~99%          | \~70%         | Compact but overfit; poor test performance.                |

---

## 🔍 Key Takeaways

* **MLPs** are quick to train but miss spatial features.
* **Basic CNNs** achieve excellent accuracy with minimal compute.
* **Dropout** helps regularization but isn’t critical here.
* **Transfer learning** requires careful preprocessing and fine-tuning to avoid underfitting.
* **MobileNetV2** overfits without augmentation, hurting generalization.

---

## 🎯 Final Results

* **Best Model:** CNN with 2 Conv layers + MaxPooling
* **Test Accuracy:** \~99%
* **Training Time:** < 10 epochs with early stopping
* **Worst Model:** ResNet50V2 (\~2–3% test accuracy due to underfitting)

---

## 🚀 How to Use

1. Clone or download the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Place the extracted GTSRB dataset inside a `gtsrb/` folder.
4. Run training:

   ```bash
   python traffic.py
   ```
5. Evaluate saved models or visualize predictions using built-in functions.

---

## 📊 Visualizations

* Training & validation accuracy/loss curves using `matplotlib`
* Prediction previews showing actual vs predicted class
* Model summary and saved weights using callbacks

---

## 📚 References & Tools

* [TensorFlow/Keras](https://www.tensorflow.org/) — Deep learning models
* [OpenCV](https://opencv.org/) — Image loading and processing
* [NumPy](https://numpy.org/) — Numerical arrays and manipulation
* [Matplotlib](https://matplotlib.org/) — Visualization
* [pandas](https://pandas.pydata.org/) — Data handling and analysis
* [scikit-learn](https://scikit-learn.org/) — Data splitting and metrics
* [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) — Benchmark dataset for traffic signs

---

## ⚖️ License

This project is licensed under the **MIT License**.
