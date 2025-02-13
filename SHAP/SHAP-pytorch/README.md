# SHAP (Shapley Additive Explanations)
📢 PyTorch Implementation Based on the Original Paper
This project is based on the paper "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
and implements SHAP using PyTorch from scratch.
<br/>
<br/>
## 1. What is SHAP?
SHAP (Shapley Additive Explanations) is a game-theoretic approach to explain the output of any machine learning model.
It quantifies the contribution of each input feature to the final prediction.

### ✅ Key Concepts of SHAP
  - Evaluates the model's output changes by adding or removing features.
  - Fairly distributes the model's prediction among individual feature contributions.
  - Considers all possible feature combinations to compare the average contribution of each feature.

### 💡 Why SHAP? <br/>
- Provides consistent and fair feature attribution based on Shapley values. Works for both classification and regression models. Offers global (entire dataset) and local (single prediction) explanations.

### 🌟 Key Principles <br/>
1️⃣ Perturbation → Generate variations of the original input by removing/altering features.<br/>
2️⃣ Shapley Value Calculation → Compute the marginal contribution of each feature using coalitional game theory.<br/>
3️⃣ Local Explanations → Aggregate contributions to explain a model’s decision for a specific instance.
<br/>
<br/>
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
<br/>

## 3. Project Directory Path

```bash
SHAP-pytorch/
│── README.md                 
│── requirements.txt
│── main.py
│── notebooks/     
│── src/                      
│   │── explain.py
│   │── model.py          
│   │── shap.py
│   │── train.py                
│   │── utils.py              
│── results/
```
<br/>

## 4. Key Implementation Steps

The SHAP values, as proposed in the original paper, are defined using the **Shapley Value** equation:

$\[
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left( f(S \cup \{i\}) - f(S) \right)
\]$

### 📌 Explanation of the Equation

- **$\phi_i$** : The SHAP value for a specific feature $i$  
- **$F$** : The set of all features  
- **$S$** : A subset of features excluding $i$  
- **$f(S)$** : The model's prediction when only the features in $S$ are present  
- **$f(S \cup \{i\})$** : The model's prediction when feature $i$ is added to $S$  
  

### 🔍 Key Concept

The SHAP value measures the contribution of feature **$i$** by evaluating the difference in model predictions when **$i$** is present vs. absent.  
It averages this difference over all possible feature subsets **$S$**, weighted according to the number of features.


✅ **(1) Define the Model**
  - Create a `simple MLP` model to generate predictions.
  - The model should be compatible with SHAP explanations.
    
✅ **(2) Generate Perturbations**
  - Randomly modify the input feature values `(perturbation)` to estimate their impact.
  - This is a practical approach instead of computing all feature subsests explicitly.

✅ **(3) Compute Weights for Perturbations**
  - Assign `higher weights` to perturbed samples that are `closer to the original instance` using `RBF kernel`.
  - This follows the SHAP principle of weighting subsets based on their probability of occurrernce.
    
✅ **(4) Compute SHAP Values**
  - Compute the baseline prediction (when all features are removed).
  - Compute the perturbed predictions.
  - Calculate the difference between the perturbed and baseline predictions.
  - Weight these differences to obtain SHAP values.

### 🌟 Mapping Equations to code

SHAP (SHapley Additive exPlanations) values are computed using the Shapley value formula:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left( f(S \cup \{i\}) - f(S) \right)
$$

This formula can be mapped to the corresponding **PyTorch implementation** as follows:

| **SHAP Formula** | **Code Implementation** |
|-----------------|----------------------|
| $\phi_i$ (SHAP value) | `shap_values[i]` |
| $S \subseteq F \setminus \{i\}$ (Subset excluding $i$) | `perturbations` (randomly masked features) |
| $f(S)$ (Model prediction for subset $S$) | `f_S = self.model(perturbations).detach()` |
| $f(S \cup \{i\})$ (Model prediction including feature $i$) | `f_S_i = self.model(perturbed_with_i).detach()` |
| Shapley Weight | `compute_weights()` function |

<br/>

## 5. Results & Analysis
### 🔹 Experiment Results
- **Positive (+) SHAP Values:**
  - These features increase the model’s prediction when present.
For example, Feature Index 10, 16, 18 play a significant role in boosting the prediction value.

- **Negative (-) SHAP Values:**
  - These features lower the model’s prediction when present.
Particularly, Feature Index 4, 6, 9, 13 strongly contribute to decreasing the predicted outcome.
This means that when these features are present, the model is more likely to predict a lower probability for a specific class.

- **Distribution of Feature Contributions:**
  - Larger SHAP values suggest that the model is highly sensitive to those features.
Features 10, 16, and 18 positively impact the prediction, while 4, 6, and 9 negatively impact it.
On the other hand, Feature Index 7, 14, and 19 seem to have minimal influence.


- **Possible Model Bias or Overfitting:**
  - Some features (Index 6, 9, 13, etc.) exhibit extreme SHAP values, which might indicate overfitting or bias toward certain features.
If SHAP values are highly asymmetrical, the model may be overly dependent on a few features, which can be problematic.


### 🔹 Feature Importance Graph Example

![image](https://github.com/user-attachments/assets/d5754b52-9d7a-4a77-a279-1a7ab207054f)

<br/>

## 6. Next Steps 🚀
✅ Compare SHAP with LIME to analyze performance differences.
✅ Apply SHAP to image/text datasets and evaluate its interpretability.
✅ Optimize SHAP computation by reducing the number of perturbations while maintaining accuracy.
✅ Explore other explainability methods (LRP, Integrated Gradients) for comparison.

<br/>

## 7. References 📚
📄 **Paper:** "A Unified Approach to Interpreting Model Predictions" – Lundberg & Lee (2017)
🔗 **Official SHAP Library:** slundberg/shap
📑 **Shapley Value Theory:** Lloyd S. Shapley (1953) – Contributions to the Theory of Games

💡 Feedback and contributions to this project are always welcome! 🚀
