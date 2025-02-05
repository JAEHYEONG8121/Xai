# LIME (Local Interpretable Model-Agnostic Explanations)

📢 **PyTorch Implementation Based on the Original Paper**  
This project is based on the paper ["Why Should I Trust You?" (Ribeiro et al., 2016)](https://arxiv.org/abs/1602.04938)  
and implements LIME using **PyTorch** from scratch.

## 1. What is LIME?
LIME (Local Interpretable Model-Agnostic Explanations) is a method for **explaining the predictions of machine learning models**.  
It can be applied to any model and helps interpret its predictions.

**🌟 Key Principles**  
1️⃣ **Perturbation** → Modify the original data to generate new samples.  
2️⃣ **Weighting** → Assign higher weights to samples closer to the original data.  
3️⃣ **Local Model Training** → Train a linear regression model to improve interpretability.  

## 2. Quick Start
### 1️⃣ **Set Up Virtual Environment & Install Dependencies (Anaconda)**
```bash
# Create a new virtual environment
conda create --name lime_env python=3.8

# Activate the environment
conda activate lime_env

# Install required packages
pip install -r requirements.txt
```

## 3. Project Directory Path

LIME-PyTorch/
│── README.md                 
│── requirements.txt
│── main.py
│── lime_experiment.ipynb     
│── src/                      
│   │── model.py              
│   │── lime.py               
│   │── utils.py              
│── results/                  
│   └── feature_importance.png
