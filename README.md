# **README: High-Risk Patient Identification and Insurance Underwriting Optimization**  

## **📌 Context and Business Case**  
In our project, we aim to develop a **machine learning model** to accurately identify **high-risk patients** who are more likely to require hospitalization. This is critical for **healthcare providers** and **insurance companies**, as early identification enables **proactive intervention**, optimized **resource allocation**, and improved **risk assessment** for policy underwriting.  

Our team has explored **multiple classification models**, focusing on **recall optimization** to ensure that we **capture as many high-risk patients as possible** while maintaining a **balanced tradeoff with precision**. The goal is to **reduce false negatives** (patients incorrectly classified as low risk) while keeping **false positives manageable** to avoid unnecessary financial or medical interventions.

---

## **📊 Key Results and Model Selection**  

### **1️⃣ Best Performing Model: CatBoost with Threshold Optimization**  
After extensive experimentation, **CatBoost** emerged as the best tree-based model for recall. We further optimized it by tuning `scale_pos_weight` and adjusting the **classification threshold** using **AUC-ROC and Youden’s J statistic**.

| **Metric** | **Value** | **Implication** |
|------------|----------|----------------|
| **AUC-ROC Score** | 0.8314 | The model effectively distinguishes between high and low-risk patients. |
| **Optimal Threshold** | 0.542 | Balances recall and false positive rate for risk assessment. |
| **Recall (High-Risk Patients)** | 67% | Captures **most** high-risk cases, reducing false negatives. |
| **Precision (High-Risk Patients)** | 33% | Acceptable tradeoff given the importance of recall. |
| **Overall Accuracy** | 79% | The model maintains a strong balance across all predictions. |

📌 **Final Decision:** We selected the **recall-optimized CatBoost model (Threshold = 0.542)** for **risk assessment and insurance underwriting**. This ensures that **more at-risk patients are detected**, reducing potential **financial losses and medical risks**.

---

## **📁 Project Structure**  
```
/project_root/
│── 📂 data/                # Processed datasets
│── 📂 models/              # Trained model files
│── 📂 notebooks/           # Core Jupyter notebooks
│── 📂 src/                 # Python scripts for data processing & model training
│── 📂 misc/                # Additional experiments & exploratory analysis
│── 📂 reports/             # Model evaluation results
│── README.md               # This file
│── requirements.txt        # Dependencies for the project
│── main.py                 # Main execution script
```

---

## **💡 Next Steps and Future Improvements**
- **Feature Engineering Enhancements**: Explore **new patient risk factors** to improve predictions.
- **Explainability & Interpretability**: Use **SHAP values** to better understand model decisions.
- **Threshold Fine-Tuning**: Evaluate **business-specific cost-benefit analysis** to adjust the **optimal recall-precision balance**.
- **Deployment and Integration**: Deploy the model into **real-world healthcare and insurance systems**.
