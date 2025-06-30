# AI Traffic Sign Classification (Deep Neural Networks & Computer Vision) 

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![NumPy](https://img.shields.io/badge/numpy-1.26.4-blueviolet)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-red)
![pandas](https://img.shields.io/badge/pandas-2.2.2-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-yellowgreen)

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
* 
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

## 🔍 Key Takeaways on this project

* **MLPs** are quick to train but miss spatial features.
* **Basic CNNs** achieve excellent accuracy with minimal compute.
* **Dropout** helps regularization but isn’t critical here.
* *Pre-trained models* like **ResNet50V2** and **MobileNetV2** overfits without augmentation, hurting generalization. They might be great for more complex projects, but the scope of this project is more suited for basic **Convulational Neural Network**.

---

## 🎯 Final Results
* **Best Model:** CNN with 2 Conv layers + MaxPooling
* **Test Accuracy:** \~99%

---

## 🚀 How to Use

1. Clone or download the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
---

## 📚 References & Tools

* [TensorFlow/Keras](https://www.tensorflow.org/) — Deep Learning Neural Network models
* [OpenCV](https://opencv.org/) — Image loading and processing
* [NumPy](https://numpy.org/) — Numerical arrays and manipulation
* [Matplotlib](https://matplotlib.org/) — Data Visualization
* [pandas](https://pandas.pydata.org/) — Data handling and analysis
* [scikit-learn](https://scikit-learn.org/) — Data splitting and metrics
* [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) — Benchmark dataset for traffic signs

---
