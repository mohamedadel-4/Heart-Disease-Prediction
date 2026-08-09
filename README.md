# Heart Disease Prediction

A Streamlit interface for the Decision Tree heart-disease classification model from the provided notebook.

## Model
- Decision Tree Classifier
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation
- Weighted F1 scoring
- Same preprocessing sequence used in the notebook

## Test-set results
- Accuracy: 0.8000
- Weighted F1: 0.8004
- ROC-AUC: 0.8461

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the Streamlit app automatically.

## Files
- `app.py` - Streamlit interface
- `final_model.pkl` - trained Decision Tree
- `encoder.pkl` - OneHotEncoder
- `label_encoder.pkl` - target LabelEncoder
- `imputer.pkl` - preprocessing imputer
- `feature_config.json` - input configuration
- `train_data.csv` / `test_data.csv` - datasets
