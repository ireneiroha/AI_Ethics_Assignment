# 📚 AI Ethics Group Assignment

**Course:** AI Ethics | PLP Academy | 2025

---

## 👥 **Team Roles and Deliverables**

| Team Member | Role | Deliverables |
| ---------------- | ---------------- | ----------------------------- |
| **1** | Theoretical Understanding | Definitions, principles, GDPR |
| **2** | Case Study — Amazon Hiring Tool | Bias source, fixes, metrics |
| **3** | Case Study — Facial Recognition | Risks, responsible policies |
| **4** | Practical Audit | Python audit using AIF360 |
| **5** | Reflection + Policy | 300-word reflection + healthcare AI policy |

---

## 📌 **1️⃣ Theoretical Understanding**

**Q1: What is Algorithmic Bias?**  
> Algorithmic bias occurs when an AI system produces systematically prejudiced results due to erroneous assumptions in the machine learning process.  
> **Examples:**  
> - A resume screening tool favoring male candidates due to biased training data.  
> - A credit scoring system assigning lower scores to minorities due to historical inequities.

**Q2: Transparency vs. Explainability**  
- **Transparency:** Openness about how the AI system is built, trained, and how its data flows.  
- **Explainability:** Ability to interpret *why* the AI makes a certain prediction.  
**Importance:** Transparency builds trust. Explainability ensures accountability and correction of harmful outcomes.

**Q3: GDPR’s Impact on AI**  
- Requires explicit user consent for personal data use.  
- Imposes the *right to explanation* for automated decisions.  
- Promotes data minimization and privacy-by-design, which limits intrusive AI.

**Ethical Principles:**  
| Principle | Definition |
|-----------------|-------------------------------|
| A) Justice | Fair distribution of AI benefits and risks |
| B) Non-maleficence | Ensure AI does no harm |
| C) Autonomy | Respect for user control and consent |
| D) Sustainability | AI design that protects the environment |

---

## 📌 **2️⃣ Case Study — Amazon Hiring Tool**

- **Source of Bias:**  
  Historical resume data was male-dominated → model penalized resumes containing words linked to women’s roles or colleges.
  
- **Fairness Fixes:**  
  1. Use balanced training data with equal gender representation.  
  2. Remove gender-related keywords.  
  3. Apply fairness-aware algorithms like reweighing or adversarial debiasing.

- **Metrics Used:**  
  - Disparate Impact Ratio  
  - Equal Opportunity Difference  
  - Demographic Parity

---

## 📌 **3️⃣ Case Study — Facial Recognition**

- **Ethical Risks:**  
  - Racial profiling: High false positives for minorities → wrongful arrests.  
  - Mass surveillance: Used without consent → civil liberties erosion.

- **Responsible Deployment Policies:**  
  - Mandatory human verification before actions.  
  - Bias testing before deployment.  
  - Restrict use to high-risk cases with legal oversight.  
  - Ensure independent audits and accountability.

---

## 📌 **4️⃣ Practical Audit — COMPAS Dataset**

- **Steps:**  
  1. Loaded `compas-scores-two-years.csv` locally.  
  2. Encoded `race_binary` to compare African-American vs Caucasian.  
  3. Used `BinaryLabelDataset` from AIF360.  
  4. Calculated Statistical Parity Difference, Equal Opportunity Difference, False Positive Rate by race.  
  5. Applied **Reweighing** to mitigate bias.  
  6. Visualized False Positive Rates with a bar chart.

- **Key Finding:**  
  *Example:* The audit revealed a higher FPR for African-Americans compared to Caucasians. After reweighing, the disparity reduced significantly, demonstrating the effectiveness of bias mitigation.

---

## 📌 **5️⃣ Reflection + Policy Proposal**

**Reflection:**  
> *In a recent project predicting student dropouts, I prioritized fairness by anonymizing socio-economic indicators and using stratified sampling. Going forward, I will apply bias detection tools (e.g., SHAP) and include affected groups in design validation. Ensuring autonomy and justice will guide my choices in data sourcing, model design, and deployment.*

**Policy — Ethical AI Use in Healthcare**

1. **Consent Protocols:**  
   - Obtain explicit informed consent for patient data.
   - Allow patients to opt-out of AI-driven decisions.

2. **Bias Mitigation Strategies:**  
   - Use diverse datasets.
   - Regular audits using fairness metrics.
   - Involve clinicians and ethicists in model development.

3. **Transparency Requirements:**  
   - Provide explainable outputs to patients and practitioners.
   - Document model design and risks.
   - Publish audit reports for accountability.

---

## ✅ **How to Run the Practical Audit**

1. Place `compas-scores-two-years.csv` in the same folder as `fairness_Team4_COMPAS_Audit_Code_FIXED.py`.  
2. Install requirements:  
   ```bash
   pip install pandas matplotlib scikit-learn aif360
   
3. Run the script:
    python fairness_Team4_COMPAS_Audit_Code_FIXED.py
   
5. Review the bias metrics and chart output.

📚 References
ProPublica COMPAS Dataset

IBM AI Fairness 360 Toolkit

EU AI Guidelines

OECD AI Principles
