# Breast Cancer Patient Insights

An interactive Streamlit dashboard for exploring a de-identified breast-cancer dataset.
It summarizes the dataset's demographic, tumor, stage, receptor, and recorded survival
fields. Results are descriptive associations, not clinical advice, diagnosis, or treatment
recommendations.

## Run locally

From the project directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

`Breast_Cancer.csv` is loaded from the project directory by default. A replacement CSV
can be selected in the app sidebar; it must contain the same columns as the included file.

The app strips whitespace from column names and text values, normalizes the Grade IV label,
and removes exact duplicate records. Tumor-size outlier exclusion (the 1.5×IQR rule used in
the accompanying analysis) is optional and disabled by default.
