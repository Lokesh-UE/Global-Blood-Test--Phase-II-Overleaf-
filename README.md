# Health Risk Prediction from Blood Test Data

## Author
**Lokesh Chamakuri**  
MSc Program, University of Europe for Applied Sciences  
Supervisor: Prof. Dr. Raja Hashim Ali

---

## Project Overview

This project develops an ensemble machine learning framework for predicting patient health risks from routine blood test data. The system integrates multiple biomarker groups (hematological, metabolic, lipid, inflammatory, vital signs) through stacking fusion with cross-feature interaction modeling and explainable AI techniques (SHAP and LIME).

**Key Results:**
- **95.8% Accuracy** and **0.987 ROC-AUC** on the Global Blood Test Health Insights 2025-2026 dataset
- **13.4% improvement** over single-feature-group baselines
- Strong robustness against noise, missing values, and adversarial perturbations
- Interpretable risk alerts with 0.96 fidelity and 0.91 stability

---

## Dataset

**Source:** [Kaggle - Global Blood Test Health Insights 2025-2026](https://www.kaggle.com/datasets/kantesti/global-blood-test-health-insights-2025-2026)

| Property | Value |
|----------|-------|
| Size | 52,000+ samples |
| Features | 21 biomarker + demographic variables |
| Target | High_Risk (binary), Risk_Category (multi-class) |
| Format | CSV |

**Feature Groups:**
- Hematological: Hemoglobin, WBC, Platelet, RBC, MCV
- Metabolic: Glucose, BMI
- Lipid Profile: Total Cholesterol, HDL, LDL
- Inflammatory: CRP, Ferritin
- Vital Signs: Systolic BP, Diastolic BP
- Demographics: Age, Gender, Region

---

## Repository Structure
