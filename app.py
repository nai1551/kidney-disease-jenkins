# -*- coding: utf-8 -*-
"""
Kidney Disease Prediction System - Streamlit App

Converted from the uploaded Gradio/Colab capstone script.
Place these files in the same Streamlit project folder:
    app.py
    kidney_disease_realistic_fixed.csv
    saved_models/
        best_model.pkl
        scaler.pkl
        label_encoders.pkl
        metadata.pkl

If the saved model files are missing, use the "Train / rebuild model" option
in the sidebar. Training can take several minutes.
"""

import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional explanation libraries
try:
    import shap
except Exception:
    shap = None

try:
    import lime
    import lime.lime_tabular
except Exception:
    lime = None


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "kidney_disease_realistic_fixed.csv"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🏥",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================
@st.cache_resource
def load_saved_artifacts():
    required = [
        MODEL_DIR / "best_model.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "label_encoders.pkl",
        MODEL_DIR / "metadata.pkl",
    ]

    if not all(p.exists() for p in required):
        return None

    with open(required[0], "rb") as f:
        best_model = pickle.load(f)
    with open(required[1], "rb") as f:
        scaler = pickle.load(f)
    with open(required[2], "rb") as f:
        label_encoders = pickle.load(f)
    with open(required[3], "rb") as f:
        metadata = pickle.load(f)

    return best_model, scaler, label_encoders, metadata


def train_and_save_model():
    """Train the same pipeline used in the uploaded capstone script."""
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        VotingClassifier,
        StackingClassifier,
        AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
    )
    from xgboost import XGBClassifier
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. "
            "Upload/copy the CSV into the Streamlit project folder."
        )

    progress = st.progress(0)
    status = st.empty()

    status.write("📥 Loading dataset...")
    data = pd.read_csv(DATA_PATH, encoding="latin-1")
    data = data.dropna()
    progress.progress(5)

    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    class_names = {
        i: cls for i, cls in enumerate(label_encoders["Target"].classes_)
    }

    target_col = "Target"
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    num_classes = y.nunique()

    status.write("⚖️ Applying SMOTE-Tomek...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smotetomek = SMOTETomek(
        smote=SMOTE(random_state=42),
        tomek=TomekLinks(sampling_strategy="auto"),
        random_state=42,
    )
    X_resampled, y_resampled = smotetomek.fit_resample(X_scaled, y)
    progress.progress(15)

    status.write("✂️ Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled,
    )

    status.write("🔎 Selecting top 10 features with Chi-Square...")
    train_min = X_train.min(axis=0)
    X_train_nonneg = X_train - train_min
    X_test_nonneg = X_test - train_min

    selector = SelectKBest(chi2, k=10)
    selector.fit(X_train_nonneg, y_train)

    selected_idx = selector.get_support(indices=True)
    selected_features = [X.columns[i] for i in selected_idx]

    X_train_selected = X_train[:, selected_idx]
    X_test_selected = X_test[:, selected_idx]
    progress.progress(25)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    tuned_models = {}
    best_params = {}
    cv_scores = {}

    model_specs = [
        (
            "Random Forest",
            RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight=class_weight_dict
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [15, 25],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 4],
            },
            8,
            5,
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
            4,
            3,
        ),
        (
            "XGBoost",
            XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 150],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            4,
            3,
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(
                random_state=42, n_jobs=-1, class_weight=class_weight_dict
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [15, 25],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 4],
            },
            8,
            5,
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(
                random_state=42, class_weight=class_weight_dict
            ),
            {
                "max_depth": [10, 15, 20],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 4],
                "max_features": ["sqrt", "log2"],
            },
            8,
            5,
        ),
        (
            "Logistic Regression",
            LogisticRegression(
                random_state=42,
                multi_class="ovr",
                class_weight=class_weight_dict,
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear", "saga"],
                "max_iter": [1000, 2000],
            },
            8,
            5,
        ),
        (
            "KNN",
            KNeighborsClassifier(n_jobs=-1),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            8,
            5,
        ),
        (
            "AdaBoost",
            AdaBoostClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.5, 1.0, 1.5],
            },
            8,
            5,
        ),
    ]

    for i, (name, estimator, params, n_iter, cv) in enumerate(model_specs, start=1):
        status.write(f"🤖 Tuning {name} ({i}/{len(model_specs)})...")
        search = RandomizedSearchCV(
            estimator,
            params,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train_selected, y_train)
        tuned_models[name] = search.best_estimator_
        best_params[name] = search.best_params_
        cv_scores[name] = search.best_score_
        progress.progress(25 + int(i * 5))

    top_3_names = sorted(cv_scores, key=cv_scores.get, reverse=True)[:3]

    status.write(f"🧩 Building ensembles from: {', '.join(top_3_names)}")
    voting_clf = VotingClassifier(
        estimators=[(n, tuned_models[n]) for n in top_3_names],
        voting="soft",
        n_jobs=-1,
    )
    voting_clf.fit(X_train_selected, y_train)

    stacking_clf = StackingClassifier(
        estimators=[(n, tuned_models[n]) for n in top_3_names],
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        cv=3,
        n_jobs=-1,
    )
    stacking_clf.fit(X_train_selected, y_train)

    all_models = {
        **tuned_models,
        "Voting Ensemble": voting_clf,
        "Stacking Ensemble": stacking_clf,
    }

    results = {}
    for name, model in all_models.items():
        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="macro"
            ),
            "PR-AUC": average_precision_score(
                pd.get_dummies(y_test), y_prob, average="macro"
            ),
            "Precision": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
        }

    results_df = pd.DataFrame(results).T.sort_values(
        "ROC-AUC", ascending=False
    )

    best_model_name = results_df.index[0]
    best_model = all_models[best_model_name]

    metadata = {
        "selected_features": selected_features,
        "selected_idx": selected_idx,
        "feature_names": list(X.columns),
        "num_classes": num_classes,
        "class_names": class_names,
        "best_model_name": best_model_name,
        "all_results": results,
        "best_params": best_params,
    }

    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open(MODEL_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    results_df.to_csv(MODEL_DIR / "model_comparison.csv")

    progress.progress(100)
    status.success(f"✅ Model ready: {best_model_name}")

    # Clear cached artifact state so the newly trained files are loaded.
    load_saved_artifacts.clear()
    return best_model, scaler, label_encoders, metadata


def calculate_egfr(age, serum_creatinine, gender="male"):
    """CKD-EPI-style calculation preserved from the original app."""
    if gender == "female":
        kappa, alpha, sex_factor = 0.7, -0.329, 1.018
    else:
        kappa, alpha, sex_factor = 0.9, -0.411, 1.0

    scr_kappa = serum_creatinine / kappa
    if scr_kappa < 1:
        egfr = 141 * (scr_kappa ** alpha) * (0.9929 ** age) * sex_factor
    else:
        egfr = 141 * (scr_kappa ** -1.209) * (0.9929 ** age) * sex_factor

    return round(max(1.0, min(120.0, egfr)), 1)


def safe_encode(label_encoders, column, value, default_text=None):
    """Encode categorical input while avoiding a crash on unexpected values."""
    le = label_encoders[column]
    candidate = value if value else default_text

    if candidate in le.classes_:
        return le.transform([candidate])[0]

    # Case-insensitive fallback
    lookup = {str(x).lower(): x for x in le.classes_}
    key = str(candidate).lower()
    if key in lookup:
        return le.transform([lookup[key]])[0]

    return le.transform([le.classes_[0]])[0]


def make_prediction(
    age,
    gender,
    albumin_urine,
    bacteria,
    blood_urea,
    serum_creatinine,
    hemoglobin,
    hypertension,
    appetite,
    anemia,
    serum_albumin,
    best_model,
    scaler,
    label_encoders,
    metadata,
):
    feature_names = metadata["feature_names"]
    selected_idx = metadata["selected_idx"]
    selected_features = metadata["selected_features"]
    class_names = metadata["class_names"]

    egfr = calculate_egfr(int(age), float(serum_creatinine), gender.lower())

    bacteria_enc = safe_encode(
        label_encoders, "Bacteria in urine", bacteria, "not present"
    )
    hypert_enc = safe_encode(
        label_encoders, "Hypertension (yes/no)", hypertension, "no"
    )
    appetite_enc = safe_encode(
        label_encoders, "Appetite (good/poor)", appetite, "good"
    )
    anemia_enc = safe_encode(
        label_encoders, "Anemia (yes/no)", anemia, "no"
    )

    # Same baseline values as the uploaded Gradio application.
    normal_baseline = {
        "Age of the patient": 35,
        "Blood pressure (mm/Hg)": 80,
        "Specific gravity of urine": 1.020,
        "Albumin in urine": 0,
        "Sugar in urine": 0,
        "Red blood cells in urine": 1,
        "Pus cells in urine": 1,
        "Pus cell clumps in urine": 0,
        "Bacteria in urine": 0,
        "Random blood glucose level (mg/dl)": 100,
        "Blood urea (mg/dl)": 20,
        "Serum creatinine (mg/dl)": 1.0,
        "Sodium level (mEq/L)": 140,
        "Potassium level (mEq/L)": 4.0,
        "Hemoglobin level (gms)": 14,
        "Packed cell volume (%)": 44,
        "White blood cell count (cells/cumm)": 8000,
        "Red blood cell count (millions/cumm)": 5.0,
        "Hypertension (yes/no)": 0,
        "Diabetes mellitus (yes/no)": 0,
        "Coronary artery disease (yes/no)": 0,
        "Appetite (good/poor)": 0,
        "Pedal edema (yes/no)": 0,
        "Anemia (yes/no)": 0,
        "Estimated Glomerular Filtration Rate (eGFR)": 100,
        "Urine protein-to-creatinine ratio": 0.1,
        "Urine output (ml/day)": 1500,
        "Serum albumin level": 4.5,
        "Cholesterol level": 180,
        "Parathyroid hormone (PTH) level": 40,
        "Serum calcium level": 9.5,
        "Serum phosphate level": 3.5,
        "Family history of chronic kidney disease": 0,
        "Smoking status": 0,
        "Body Mass Index (BMI)": 22,
        "Physical activity level": 1,
        "Duration of diabetes mellitus (years)": 0,
        "Duration of hypertension (years)": 0,
        "Cystatin C level": 0.9,
        "Urinary sediment microscopy results": 0,
        "C-reactive protein (CRP) level": 1.0,
        "Interleukin-6 (IL-6) level": 2.0,
    }

    user_input = {
        "Albumin in urine": float(albumin_urine),
        "Bacteria in urine": float(bacteria_enc),
        "Blood urea (mg/dl)": float(blood_urea),
        "Serum creatinine (mg/dl)": float(serum_creatinine),
        "Hemoglobin level (gms)": float(hemoglobin),
        "Hypertension (yes/no)": float(hypert_enc),
        "Appetite (good/poor)": float(appetite_enc),
        "Anemia (yes/no)": float(anemia_enc),
        "Estimated Glomerular Filtration Rate (eGFR)": float(egfr),
        "Serum albumin level": float(serum_albumin),
    }

    full_vector = np.array(
        [[normal_baseline.get(col, 0) for col in feature_names]],
        dtype=float,
    )

    for feat, value in user_input.items():
        if feat in feature_names:
            full_vector[0, feature_names.index(feat)] = value

    full_scaled = scaler.transform(full_vector)
    input_final = full_scaled[:, selected_idx]

    pred_class = int(best_model.predict(input_final)[0])
    pred_proba = best_model.predict_proba(input_final)[0]
    confidence = float(pred_proba[pred_class] * 100)

    return {
        "pred_class": pred_class,
        "predicted_stage": class_names[pred_class],
        "confidence": confidence,
        "egfr": egfr,
        "probabilities": {
            class_names[i]: float(prob * 100)
            for i, prob in enumerate(pred_proba)
        },
        "input_final": input_final,
        "selected_features": selected_features,
    }


def make_shap_plot(best_model, input_final, selected_features):
    if shap is None:
        return None

    background = None
    if DATA_PATH.exists():
        # SHAP is optional. Keep the UI usable even when SHAP is unavailable.
        pass

    try:
        # KernelExplainer can be expensive. Use a small synthetic/background
        # sample around the current input to keep Streamlit responsive.
        rng = np.random.default_rng(42)
        background = np.repeat(input_final, 10, axis=0)
        background += rng.normal(0, 0.05, background.shape)

        explainer = shap.KernelExplainer(
            best_model.predict_proba,
            background,
        )
        shap_values_raw = explainer.shap_values(
            input_final,
            nsamples=100,
        )

        pred_class = int(best_model.predict(input_final)[0])

        if isinstance(shap_values_raw, list):
            shap_values_class = np.asarray(shap_values_raw[pred_class])
        elif getattr(shap_values_raw, "ndim", 0) == 3:
            # SHAP versions differ in the placement of the class dimension.
            if shap_values_raw.shape[-1] == len(selected_features):
                shap_values_class = shap_values_raw[:, pred_class, :]
            else:
                shap_values_class = shap_values_raw[:, :, pred_class]
        else:
            shap_values_class = np.asarray(shap_values_raw)

        fig = plt.figure(figsize=(9, 5))
        shap.summary_plot(
            shap_values_class,
            input_final,
            feature_names=selected_features,
            max_display=min(10, len(selected_features)),
            show=False,
        )
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Model")
st.sidebar.caption("Kidney Disease Prediction System")

artifacts = load_saved_artifacts()

if artifacts is None:
    st.sidebar.warning("Saved model files were not found.")
    if st.sidebar.button("🚀 Train / rebuild model", use_container_width=True):
        try:
            artifacts = train_and_save_model()
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))
            st.code(traceback.format_exc())
else:
    st.sidebar.success("Saved model loaded")

if artifacts is None:
    st.title("🏥 Kidney Disease Prediction System")
    st.info(
        "Add the trained model files to `saved_models/`, or place "
        "`kidney_disease_realistic_fixed.csv` beside `app.py` and train the model "
        "from the sidebar."
    )
    st.stop()

best_model, scaler, label_encoders, metadata = artifacts

st.sidebar.write(f"**Best model:** {metadata['best_model_name']}")
st.sidebar.write(f"**Selected features:** {len(metadata['selected_features'])}")

# ============================================================
# MAIN UI
# ============================================================
st.title("🏥 Kidney Disease Prediction System")
st.markdown(
    "### Enter patient information — eGFR is automatically calculated "
    "from age, gender, and serum creatinine."
)

with st.expander("⚠️ Important note", expanded=False):
    st.warning(
        "This is a machine-learning capstone/demo tool, not a medical diagnosis. "
        "Use the result for educational/research purposes and consult a qualified "
        "healthcare professional for clinical decisions."
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Info")
    age = st.number_input("Age of the patient (years)", 1, 120, 35)
    gender = st.selectbox("Gender", ["male", "female"], index=0)

    st.subheader("🧪 Urine Test")
    albumin_urine = st.number_input(
        "Albumin in urine (0–5 scale)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
    )
    bacteria = st.selectbox(
        "Bacteria in urine",
        ["present", "not present"],
        index=1,
    )

with col2:
    st.subheader("💉 Blood Tests")
    blood_urea = st.number_input(
        "Blood urea (mg/dl)", min_value=0.0, value=20.0, step=0.1
    )
    serum_creatinine = st.number_input(
        "Serum creatinine (mg/dl)", min_value=0.0, value=1.0, step=0.1
    )
    hemoglobin = st.number_input(
        "Hemoglobin level (gms)", min_value=0.0, value=14.0, step=0.1
    )
    serum_albumin = st.number_input(
        "Serum albumin level", min_value=0.0, value=4.5, step=0.1
    )

    st.subheader("📋 Clinical Findings")
    hypertension = st.selectbox(
        "Hypertension (yes/no)", ["yes", "no"], index=1
    )
    appetite = st.selectbox(
        "Appetite (good/poor)", ["good", "poor"], index=0
    )
    anemia = st.selectbox(
        "Anemia (yes/no)", ["yes", "no"], index=1
    )

st.divider()

egfr_preview = calculate_egfr(age, serum_creatinine, gender)
st.metric("Auto-calculated eGFR", f"{egfr_preview} ml/min/1.73m²")

predict = st.button(
    "🔍 Predict & Explain",
    type="primary",
    use_container_width=True,
)

if predict:
    try:
        output = make_prediction(
            age,
            gender,
            albumin_urine,
            bacteria,
            blood_urea,
            serum_creatinine,
            hemoglobin,
            hypertension,
            appetite,
            anemia,
            serum_albumin,
            best_model,
            scaler,
            label_encoders,
            metadata,
        )

        st.success(
            f"Predicted Stage: **{output['predicted_stage']}**"
        )

        m1, m2 = st.columns(2)
        m1.metric("Prediction confidence", f"{output['confidence']:.2f}%")
        m2.metric(
            "eGFR",
            f"{output['egfr']} ml/min/1.73m²",
        )

        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame(
            {
                "Class": list(output["probabilities"].keys()),
                "Probability (%)": list(output["probabilities"].values()),
            }
        ).set_index("Class")
        st.bar_chart(prob_df)

        st.subheader("🧠 SHAP Explanation")
        shap_fig = make_shap_plot(
            best_model,
            output["input_final"],
            output["selected_features"],
        )
        if shap_fig is not None:
            st.pyplot(shap_fig, clear_figure=True)
        else:
            st.info(
                "SHAP explanation could not be generated. "
                "The prediction itself is still available."
            )

        st.subheader("📌 Model Information")
        st.write(
            f"**Best model:** {metadata['best_model_name']}  \n"
            f"**Top features:** {', '.join(output['selected_features'])}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


# ============================================================
# MODEL COMPARISON
# ============================================================
with st.expander("📈 Model comparison"):
    results = metadata.get("all_results", {})
    if results:
        comparison_df = (
            pd.DataFrame(results)
            .T.sort_values("ROC-AUC", ascending=False)
        )
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("No saved comparison data found.")

st.caption(
    "Kidney Disease Prediction System • Streamlit deployment version"
)
