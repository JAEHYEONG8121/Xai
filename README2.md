## SHAP (Shapley Additive Explanations)
📢 PyTorch Implementation Based on the Original Paper
This project is based on the paper "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
and implements SHAP using PyTorch from scratch.

### 1. What is SHAP?
SHAP (Shapley Additive Explanations) is a game-theoretic approach to explain the output of any machine learning model.
It quantifies the contribution of each input feature to the final prediction.

**✅ Key Concepts of SHAP**
  - Evaluates the model's output changes by adding or removing features.
  - Fairly distributes the model's prediction among individual feature contributions.
  - Considers all possible feature combinations to compare the average contribution of each feature.

**💡 Why SHAP?** <br/>
- Provides consistent and fair feature attribution based on Shapley values. Works for both classification and regression models. Offers global (entire dataset) and local (single prediction) explanations.

**🌟 Key Principles** <br/>
1️⃣ Perturbation → Generate variations of the original input by removing/altering features.<br/>
2️⃣ Shapley Value Calculation → Compute the marginal contribution of each feature using coalitional game theory.<br/>
3️⃣ Local Explanations → Aggregate contributions to explain a model’s decision for a specific instance.
