# **A Novel Growth Pattern-Aware Digital Marker of TILs Improves Stratification of Lung Adenocarcinoma Patients**

This repositry contains the code for the following manuscript: 

## **Abstract**
Lung adenocarcinoma (LUAD) is one of the most prevalent forms of cancer and continues to be associated with high mortality rates, despite recent advances in cancer therapy. Effective risk stratification is critical for guiding treatment decisions and improving our understanding of disease mechanisms. However, current prognostic approaches face considerable limitations. Growth pattern-based grading serves as prognostic indicator of tumour aggressiveness but is inherently subjective and prone to high degree of variability among observers. Other well-established prognosis indicators such as tumour infiltrating lymphocytes (TILs) and stromal TILs (sTILs) scores, provide valuable prognostic information but require labour-intensive assessment. The pronounced heterogeneity of LUAD further complicates prognosis and underscores the need for robust, integrative biomarkers that capture both the morphological and immunological characteristics of tumour. To address this need, we propose an AI based Growth Pattern Specific TILs (GPS-TILs) marker, that quantifies TILs and sTILs within each growth pattern separately. By integrating morphological information from the patterns and immune microenvironment information of TILs, we show that the proposed GPS-TILs enhance patient stratification. We evaluated the prognostic utility of GPS-TILs using survival analysis with Cox proportional hazards models in a cross-validation setting on The Cancer Genome Atlas LUAD (TCGA-LUAD) cohort. Our findings reveal that GPS-TILs offers strong prognostic value for overall survival (p<0.0001, C-index= 0.59), outperforming conventional TIL-based measures and morphology-based stratification approaches. These results highlight the potential of GPS-TILs as a more objective and effective tool for patient risk stratification in LUAD.

## **Growth Pattern classification**
For growth pattern classification please refer to our other paper: 
https://www.sciencedirect.com/science/article/pii/S0010482525004780

## **Usage instructions**
* Grade data for TCGA-LUAD, extracted from their pathological reports, can be found under the Data folder
* For regenerating the KM-curves of various grading schemes reported in the paper use the file : Grade_KM_curves.py
* For regenrating the CV reslts of GPS-TILs use the file : Survival_CV.py
