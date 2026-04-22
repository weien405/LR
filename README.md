# AP ICU Trajectory Risk Predictor

Files required for deployment:

- `app.py`
- `requirements.txt`
- `LR.pkl`

Recommended deployment steps for Streamlit Cloud:

1. Upload these three files to your GitHub repository.
2. In Streamlit Cloud, set **Main file path** to `app.py`.
3. Redeploy or reboot the app.

Notes:

- The app is designed for the final binary Logistic Regression model only.
- It includes a compatibility patch for legacy `LogisticRegression` objects that may be missing `multi_class` after unpickling on newer scikit-learn versions.
- If prediction still fails, open the **Debug details** section in the app to inspect the traceback.
