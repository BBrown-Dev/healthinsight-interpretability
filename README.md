# HealthInsight Interpretability Toolkit

A complete, modular, and production‑ready interpretability pipeline designed to explain complex machine learning models used in healthcare. This project demonstrates how to make “black box” models transparent using feature importance, partial dependence plots (PDPs), SHAP global and local explanations, and decision boundary visualizations.

---

## Project Overview

Healthcare models often rely on complex algorithms such as ensemble methods and gradient‑boosted trees. While powerful, these models can be difficult for clinicians to trust without clear explanations.

This project simulates healthcare data, trains an XGBoost classifier, and applies multiple interpretability techniques to make model behavior understandable and actionable for healthcare professionals.

The pipeline includes:

- Feature Importance (XGBoost Gain)
- Partial Dependence Plots (PDPs)
- SHAP Global Explanations
- SHAP Local Explanations (Patient‑level)
- Decision Boundary Visualizations
- Structured Logging and Timing
- Automatic Plot Saving

All outputs are saved to the `outputs/` directory.

---

## Key Features

### 1. Data Simulation
Generates a synthetic healthcare dataset with:
- 2000 samples  
- 6 clinically inspired features  
- Balanced binary outcomes  

### 2. Model Training
Trains an XGBoost classifier and logs:
- Training accuracy  
- Training AUC  
- Model hyperparameters  

### 3. Interpretability Techniques
Includes:

#### Feature Importance  
- Ranked by XGBoost gain.

#### Partial Dependence Plots  
- Shows marginal effects of key features.

#### SHAP Global Explanations  
- Identifies the most influential features across the entire dataset.

#### SHAP Local Explanations  
- Explains the prediction for a single patient (Patient #12), including:
  - Top positive contributors  
  - Top negative contributors  
  - Saved force plot  

#### Decision Boundary Visualization  
- Shows how two features (Glucose & BMI) separate classes.

---

## Installation

Clone the repository:

```
git clone `https://github.com/BBrown-Dev/healthinsight-interpretability.git` [(github.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fgithub.com%2FBBrown-Dev%2Fhealthinsight-interpretability.git")
cd healthinsight-interpretability
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Pipeline

Execute the full interpretability workflow:

```
python main.py
```

All plots will be saved automatically to the `outputs/` directory.

---

## Example Output (Console)

```
[INFO] Dataset Summary:
    Samples: 2000
    Features: 6
    Outcome distribution: {1: 0.5005, 0: 0.4995}

[INFO] Training Accuracy: 0.9831
[INFO] Training AUC: 0.9831

[INFO] Feature Importance (sorted):
    BloodPressure: 11.4638
    Cholesterol: 9.9690
    Age: 9.9337
    HeartRate: 6.8681
    BMI: 1.3363
    Glucose: 1.2069

[INFO] Top SHAP Contributors (Global):
    BloodPressure: 1.2943
    Cholesterol: 1.2469
    Age: 1.1058
    HeartRate: 0.8103
    BMI: 0.1777

[INFO] Local SHAP Explanation for Patient #12:
    BloodPressure: 3.7557 (increases risk)
    HeartRate: 1.4184 (increases risk)
    Cholesterol: 0.7558 (increases risk)
    Age: 0.6006 (increases risk)
    BMI: -0.1742 (decreases risk)
```
---

## References
Alhamid, M. (2022, May 27). Ensemble models. Medium. https://towardsdatascience.com/ensemble-models-5a62d4f4cb0c 

Amazon Web Services. (n.d.). Model interpretability – Machine learning best practices in healthcare and life sciences. https://docs.aws.amazon.com/whitepapers/latest/ml-best-practices-healthcare-life-sciences/model-interpretability.html 

De Harder, H. (2023, July 11). Model-agnostic methods for interpreting any machine learning model. Medium. https://towardsdatascience.com/model-agnostic-methods-for-interpreting-any-machine-learning-model-4f10787ef504 

Gormely, I. (2023, June 14). Machine learning interpretability: New challenges and approaches. Vector Institute for Artificial Intelligence. https://vectorinstitute.ai/machine-learning-interpretability-new-challenges-and-approaches/ 

Hong, Z. (2023, October 6). Interpreting deep learning models: Techniques for understanding predictions. Medium. https://medium.com/@zhonghong9998/interpreting-deep-learning-models-techniques-for-understanding-predictions-470b521ec401 

Huang, A. (2020, August 24). Interpretability. Machine Learning Blog | ML@CMU. https://blog.ml.cmu.edu/2020/08/31/6-interpretability/ 

Jha, A. (2023, August 18). ML model interpretation tools: What, why, and how to interpret. Neptune.ai. https://neptune.ai/blog/ml-model-interpretation-tools 

MarkovML. (n.d.). Techniques for interpreting and explaining ML models. https://www.markovml.com/blog/model-interpreting 

Molnar, C. (2023). Interpretable machine learning. https://christophm.github.io/interpretable-ml-book/ 

Poduska, J. (2023, September 1). SHAP and LIME Python libraries: Part 1 – Great explainers, with pros and cons to both. Domino Data Lab. https://domino.ai/blog/shap-lime-python-libraries-part-1-great-explainers-pros-cons 

Princeton Dialogues on AI and Ethics. (2019, April 24). Case study PDFs. https://aiethics.princeton.edu/case-studies/case-study-pdfs/ 

Salon, D. S. (n.d.). Interpretable machine learning and how to build trust in our models. Data Science Salon. https://roundtable.datascience.salon/interpretable-machine-learning-building-trust-in-ml 