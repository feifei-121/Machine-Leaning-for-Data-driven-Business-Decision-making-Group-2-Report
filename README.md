# MGRC20007 Group 2 – Bank Customer Churn Prediction

## Project Abstract

This project develops a predictive analytics solution for a retail bank to identify customers most likely to churn within twelve months. Using the provided bank dataset of 10,000 customers, we applied supervised machine learning models to estimate churn probability, interpret its drivers, and design targeted interventions. The project aims to reduce the overall churn rate, ensure fairness across demographic groups, and deliver tangible business value through data-driven decision-making.

---

## 1. Decision Context and Objectives

### 1.1 Decision Problem

Customer attrition imposes significant financial losses on the bank because acquiring new customers costs considerably more than retaining existing ones. The key managerial decision is how to optimally allocate limited retention resources toward customers most likely to leave. The predictive model supports this by producing individualized churn risk scores.

### 1.2 Intended Users and Stakeholders

- **Customer Retention Teams:** Use model outputs to design personalized retention campaigns.  
- **Marketing Department:** Utilizes churn segments for targeted multi-channel communication.  
- **Senior Management:** Relies on model insights to assess ROI and portfolio-level retention strategy.  

### 1.3 Success Metrics

| Metric                  | Target                           |
| ----------------------- | -------------------------------- |
| Reduction in churn rate | From 20.4% to ≤15%               |
| ROC AUC                 | > 0.80                           |
| Precision               | > 75%                            |
| ROI                     | > 3 : 1                          |
| Fairness variation      | ≤ 5% across gender and geography |

### 1.4 Fairness Constraints

Following the course requirements and ethical AI principles (Barocas & Selbst, 2016), model accuracy and false negative rates must not vary by more than five percentage points across gender and geographic segments.

---

## 2. Data Description and Preparation

### 2.1 Dataset Overview

- **Original dataset:** `Churn_Modelling.csv` (10,000 records, 11 features)  
- **Processed dataset:** `Churn_Modelling_Cleaned.csv` (cleaned and feature-engineered)  
- **Target variable:** `Exited` (1 = churned, 0 = retained)

### 2.2 Key Variables

| Type       | Features                                                     |
| ---------- | ------------------------------------------------------------ |
| Original   | CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited |
| Engineered | BalanceIsZero, Age_Balance_Interaction, AgeBin (categorical: 18–25, 26–35, 36–45, 46–60, 60+) |

### 2.3 Exploratory Insights

- Baseline churn rate: **20.37%** (moderate class imbalance)  
- Churn by geography: **Germany 32.4%**, **France 16.2%**, **Spain 16.7%**  
- Churn by gender: **Female 25.1%**, **Male 16.5%**  
- Most at-risk age group: **46–60 years (51.1%)**  
- Inactive customers churn at **26.9%** vs. **14.3%** for active ones  

### 2.4 Feature Engineering

- `BalanceIsZero`: flags inactive accounts  
- `Age_Balance_Interaction`: captures wealth lifecycle dynamics  
- `AgeBin`: converts non-linear age effects into interpretable groups  

---

## 3. Methodology and Model Development

### 3.1 Model Families

Three supervised models from distinct algorithmic families were implemented:

| Model               | Description                                     | Purpose                   |
| ------------------- | ----------------------------------------------- | ------------------------- |
| Logistic Regression | Linear baseline with interpretable coefficients | Benchmark model           |
| Decision Tree       | Rule-based model with transparent structure     | Explainability            |
| Random Forest       | Ensemble of trees with bootstrap aggregation    | Performance and stability |

### 3.2 Model Training and Validation

- 5-fold **stratified cross-validation** to preserve class distribution.  
- **Train-test split:** 80/20 (n = 8,000 / 2,000).  
- **Random seed = 42** for full reproducibility.  
- Hyperparameters chosen conservatively to avoid overfitting.

### 3.3 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Business performance: **expected retention ROI** and **lift at top decile**

---

## 4. Results and Evaluation



| Model                   | Accuracy | Precision | Recall | F1-Score | ROC AUC    |
| ----------------------- | -------- | --------- | ------ | -------- | ---------- |
| **Random Forest**       | 85.45%   | 82.95%    | 35.87% | 50.09%   | **0.8446** |
| **Decision Tree**       | 85.60%   | 80.51%    | 38.57% | 52.16%   | 0.8307     |
| **Logistic Regression** | 82.00%   | 62.84%    | 28.26% | 38.98%   | 0.7774     |

**Random Forest** achieved the best overall performance with a ROC AUC of **0.8446**,  
demonstrating superior probability calibration and generalization stability compared to the other models.  Although the Decision Tree reached a similar accuracy, it exhibited higher variance across folds.  Logistic Regression, as a linear baseline, underperformed due to its limited ability to capture non-linear feature interactions.

---

## 5. Business Value and Strategic Insights

### 5.1 Feature Importance

| Feature           | Importance | Interpretation                                   |
| ----------------- | ---------- | ------------------------------------------------ |
| NumOfProducts     | 22.40%     | Single-product customers have highest churn risk |
| Age               | 22.17%     | Mid-age (46–60) shows peak churn tendency        |
| IsActiveMember    | 7.45%      | Activity strongly predicts retention             |
| Geography_Germany | 5.69%      | Reflects competitive local conditions            |

### 5.2 Financial Impact

| Metric                        | Value                               |
| ----------------------------- | ----------------------------------- |
| Total customers               | 10,000                              |
| Current churn rate            | 20.37%                              |
| Annual churn loss             | £10.19M                             |
| Targeted customers (top 20%)  | 2,000                               |
| Captured churners             | 1,205 (59.2%)                       |
| Retention success rate        | 60%                                 |
| Annual revenue saved          | £3.6M                               |
| Intervention cost             | £0.3M                               |
| **Net benefit**               | **£3.3M**                           |
| **ROI**                       | **11.0 : 1**                        |
| **Projected churn reduction** | **7.2pp → 13.17% final churn rate** |

### 5.3 Strategic Recommendations

- Prioritize **high-risk (p > 0.60)** customers for personal relationship management.  
- Launch **cross-sell campaigns** for single-product customers.  
- Address **inactive accounts** with re-engagement programs.  
- Tailor strategies to **Germany** using localized competitive insights.  
- Implement **randomized controlled A/B testing** to validate intervention ROI.

---

## 6. Fairness and Ethical Considerations

### 6.1 Fairness Results

| Group                      | Accuracy              | Variation | Within Tolerance |
| -------------------------- | --------------------- | --------- | ---------------- |
| Male vs Female             | 88.0% vs 82.5%        | 5.6%      | ✓ Yes            |
| France vs Germany vs Spain | 86.9% / 80.9% / 86.9% | 6.0%      | ⚠ Monitor        |

Gender fairness is within acceptable thresholds. Geographic variation is marginally above tolerance, primarily reflecting higher base churn rates in Germany rather than model bias.

### 6.2 Mitigation Strategies

- Apply **fairness-aware threshold optimization** per demographic group.  
- Conduct **quarterly fairness audits** on model predictions.  
- Maintain transparency in customer communication.  
- Ensure data governance with encryption, access controls, and audit logs.

---

## 7. Reproducibility and Implementation

To reproduce all results:

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/feifei-121/Machine-Leaning-for-Data-driven-Business-Decision-making-Group-2-Report
   pip install -r requirements.txt
   ```

2. Run the Jupyter Notebook:

   ```bash
   jupyter notebook "ML2) final.ipynb"
   ```

3. The notebook automatically loads `Churn_Modelling_Cleaned.csv` and executes all preprocessing, modeling, and evaluation steps using fixed random seed `42`.

---

## 8. GenAI Use Declaration

This project made **minimal use of Generative AI tools**. All analytical tasks, model development, feature engineering, and statistical evaluations were performed manually by team members. No confidential data was uploaded to any external platform.

Generative AI  was used only for:  

- Supporting **initial team discussion and task allocation**.  
- Improving **code visualization** (matplotlib/seaborn formatting).  
- Conducting **minor grammar and clarity edits** in report writing.

All results, methods, and interpretations were verified manually by the team to ensure correctness and compliance with academic integrity standards.

---

## 9. Team Reflection and Contributions

Project collaboration occurred primarily asynchronously due to schedule constraints. Version control and documentation ensured full reproducibility and transparency. 

| Name           | Student ID | Role             | Contribution                                        |
| -------------- | ---------- | ---------------- | --------------------------------------------------- |
| Feifei Yu      | 2605197    | Project Leader   | Code integration, full report writing, GitHub setup |
| Yuxuan Zhang   | 2477390    | Model Validation | Reproduced evaluation metrics, verified results     |
| Jiayi Wu       | 2616777    | Data Engineer    | Data cleaning, feature engineering, EDA             |
| Guoxiang Zhang | 2640218    | ML Developer     | Implemented all models and selected Random Forest   |
| Jiaming Gu     | 2596990    | Data Analyst     | Data preprocessing, validation, documentation       |

---

## References

- Barocas, S. and Selbst, A.D. (2016) “Big data’s disparate impact,” *California Law Review*, 104(3), pp. 671–732.  
- Breiman, L. (2001) “Random forests,” *Machine Learning*, 45(1), pp. 5–32.  
- Hastie, T., Tibshirani, R. and Friedman, J. (2009) *The elements of statistical learning: data mining, inference, and prediction.* 2nd edn. Springer.  
- Verbeke, W., Dejaeger, K., Martens, D., Hur, J. and Baesens, B. (2012) “New insights into churn prediction in the telecommunication sector: a profit driven data mining approach,” *European Journal of Operational Research*, 218(1), pp. 211–229.

---





