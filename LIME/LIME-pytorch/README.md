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
### **Set Up Virtual Environment & Install Dependencies (Anaconda)**
```bash
# Create a new virtual environment
conda create --name lime_env python=3.8

# Activate the environment
conda activate lime_env

# Install required packages
pip install -r requirements.txt
```

## 3. Project Directory Path
```bash
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
```

## 4. Key Implementation Steps

The LIME paper aims to approximate a complex model $f(x)$ with a simple, interpretable local model $g(x)$.
The optimization objective for LIME is defined as:

$$
\underset{g \in G}{\arg\min} \quad L(f, g, \pi_x) + \Omega(g)
$$

### 🌟 Explanation:
- $f(x)$ : The original ML model
- $g(x)$ : A local surrogate model (a simple linear model in LIME)
- $\pi_x$ : Weight function that gives higer importance to samples close to $x$
- $L(f, g, \pi_x)$ : Loss function, ensuring that $g(x)$ approximates $f(x)$ locally
- $\Omega(g)$ : Regularization term, ensuring that $g(x)$ remains simple and interpretable

LIME approximates $f(x)$ by generating **pertubed samples**, assigning **weights**, and training a **weighted linear regression model** to explain the prediction.

<br/>
<br/>

✅ **(1) Data Preparation & Model Traning**
- Implement the `SimpleMLP` model using PyTorch to solve a classification task.
- Generate synthetic data using `make_classification()`.

✅ **(2) Perturbation Sampling**
- Create new perturbed data samples by adding Gaussian noise to the original sample.

✅ **(3) Weighting (Applying Weights to Samples)**
- Compute distance-based weights between the original sample and perturbed samples using the RBF kernel.

✅ **(4) Training the Local Linear Model**
- Train a weighted linear regression model using the perturbed samples and weights.
<br/>
<br/>

### 🌟 Mapping Equations to code

| **Paper Equation** | **Implemented in Code (File)** | **Description** |
|-------------------|--------------------------------|------------------------------------|
| $Z = \{z_1, z_2, ..., z_n\}, z_i \sim \mathcal{N}(x, \sigma^2)$ | `generate_perturbations()`<br> (_src/utils.py_) | Generates perturbed samples using Gaussian noise |
| $\pi_x(z) = \exp\left(- \frac{d(x, z)^2}{2\sigma^2} \right)$ | `compute_weights()`<br> (_src/utils.py_) | Assigns importance weights to perturbed samples |
| $\underset{g \in G}{\arg\min} Σᵢ πₓ(zᵢ) (f(zᵢ) - g(zᵢ))² + Ω(g)$ | `train_local_model()`<br> (_src/lime.py_) | Trains the local linear model using weighted MSE loss |
| $Importance = \|wᵢ\|$| Extracted from `local_model.linear.weight`<br> (_src/lime.py_) | Computes feature importance based on absolute weight values |

## 5. Results & Analysis
### 🔹 Experiment Results
- The **feature importance** calculated by LIME for a given input dataset is visualized as a graph.
- The analysis helps to determine **which features have the most significant impact on predictions**.

### 🔹 Feature Importance Graph Example

![image](https://github.com/user-attachments/assets/65ebcc52-48f5-4c22-aa93-167c34a8c08e)

## 6. Next Steps

✅ **Compare LIME with SHAP** to analyze performance differences.<br/>
✅ **Apply LIME to image/text data** and evaluate its effectiveness.

## 7. References $ Papers

📄 **Paper**: "Why Should I Trust You?" Explaining the Predictions of Any Classifier<br/>
🔗 **Official LIME Library**: marcotcr/lime

💡 **Feedback on this project is always welcome!** 🚀













