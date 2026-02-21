# Global Mental Health Data Analysis & Policy Dashboard

An end-to-end data analysis project that explores global mental health trends using real-world datasets and presents the findings through an interactive Streamlit dashboard designed for non-technical users and policymakers.

---

##  Project Overview
Mental health conditions are common worldwide, but the countries with the highest prevalence are not always the ones experiencing the greatest real-world impact.

This project analyzes global mental health datasets to answer:

- Which countries are most affected by mental illness?
- Which disorders contribute the most globally?
- Does higher prevalence always mean higher societal burden?
- Which regions should be prioritized for mental health intervention?

To make the results accessible beyond technical audiences, the analysis was deployed as an interactive dashboard.

---

## Objectives
- Clean and merge multiple international datasets
- Analyze prevalence of major mental disorders
- Measure disease burden using DALYs (Years of Healthy Life Lost)
- Identify trends across countries and years
- Recommend priority regions for intervention programs
- Build a dashboard usable by policymakers and organizations

---

## Mental Disorders Analyzed

- Depression
- Anxiety disorders
- Bipolar disorder
- Schizophrenia
- Eating disorders

---

## Key Insights
- Higher prevalence does **not always** correspond to higher impact
- Some countries report moderate case numbers but carry heavy disease burden
- Mental health impact is strongly linked to healthcare access and awareness
- Certain countries consistently appear among the most affected over multiple years

---

## Interactive Dashboard (Streamlit)

The project includes a Streamlit web application that allows users to:

- Select a country and view mental health trends
- Compare prevalence vs disease burden
- View top affected countries
- Track changes over time

The goal was to present data in a way that **non-technical decision makers** can quickly understand and use.

### Run the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
