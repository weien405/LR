"""
Formal English Streamlit predictor for the final LR model.

Recommended minimal LR.pkl bundle schema:
{
    "model": fitted sklearn/imblearn estimator or pipeline with predict_proba,
    "feature_cols": list[str],
    "class_labels": ["Favorable", "Unfavorable"],  # optional
    "positive_label": "Unfavorable",               # optional
    "threshold": 0.5,                              # optional
    "ruleout_threshold": 0.20,                     # optional
    "rulein_threshold": 0.50,                      # optional
    "train_median": {feature: value, ...},         # optional
    "feature_meta": {                              # optional
        "vaso_any_24h": {"label_en": "...", "unit": "...", "help_en": "..."}
    },
    "example_patient": {feature: value, ...},      # optional
    "coefficient_table": [                         # optional; preferred for formal interpretation
        {
            "feature": "...",
            "coefficient": 0.12,
            "odds_ratio": 1.13,
            "direction_en": "Higher value -> higher risk"
        }
    ],
    "study_title_en": "..."
}
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Early Unfavorable Trajectory Predictor (LR)",
    page_icon="🩺",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "LR.pkl"

APP_TEXT = {
    "title": "Early Predictor for Unfavorable Trajectory in AP ICU Patients",
    "subtitle": "Research-facing web companion based on the final Logistic Regression model",
    "caption": (
        "This tool is for research demonstration and manuscript support only. "
        "It should not replace clinical judgement. "
        "Please ensure that the loaded LR.pkl exactly matches the final study model."
    ),
    "binary_negative": "Favorable trajectory",
    "binary_positive": "Unfavorable trajectory",
    "predict_button": "Run prediction",
    "fill_default": "Reset to training defaults",
    "fill_example": "Load example patient",
    "section_model": "Model overview",
    "section_input": "Patient inputs",
    "section_result": "Prediction results",
    "section_prob": "Probability output",
    "section_interpret": "Model interpretation",
    "section_snapshot": "Current input summary",
    "section_notice": "Use notes and disclaimer",
    "threshold_label": "Predefined decision threshold",
    "risk_probability": "Unfavorable probability",
    "predicted_label": "Predicted label",
    "risk_band": "Risk band",
    "download": "Download result summary CSV",
    "bundle_error": "Failed to load LR.pkl. Please verify the model file and required bundle fields.",
    "binary_model_error": "This web app is only intended for the final binary LR model.",
    "missing_model_error": "LR.pkl was not found. Please place the final LR model bundle in this folder.",
    "coef_exported_note": "The table below uses the exported formal coefficient summary from the training pipeline.",
    "coef_fallback_note": (
        "The coefficients below were extracted directly from the loaded LR model. "
        "If preprocessing includes scaling or transformations, they are best interpreted as direction only."
    ),
    "no_interpret_note": (
        "No formal coefficient_table was found in the bundle. "
        "The app will therefore limit interpretation to variables and predicted probabilities."
    ),
    "notice_text": (
        "1. Enter variables defined exactly as in the final paper; "
        "2. Default values are representative training values and should be replaced by patient-specific data; "
        "3. Recheck any value outside physiologically plausible ranges; "
        "4. This tool is intended for manuscript support, supplementary material, and online demonstration only."
    ),
    "input_help_prefix": "Variable code",
    "yes": "Yes",
    "no": "No",
    "probability_table_class": "Class",
    "probability_table_probability": "Probability",
    "snapshot_feature": "Feature",
    "snapshot_value": "Input value",
    "snapshot_unit": "Unit",
}

DEFAULT_FEATURE_META = {
    "vaso_any_24h": {
        "label_zh": "24小时内是否使用升压药",
        "label_en": "Any vasopressor within 24 h",
        "unit": "Yes/No",
        "input_type": "binary",
        "help_en": "Whether any vasopressor was used within the first 24 hours after ICU admission.",
        "default": 0,
    },
    "spo2_mean_24h": {
        "label_zh": "24小时平均 SpO₂",
        "label_en": "Mean SpO2 within 24 h",
        "unit": "%",
        "input_type": "continuous",
        "help_en": "Mean oxygen saturation during the first 24 hours after ICU admission.",
        "default": 96.0,
        "min": 50.0,
        "max": 100.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "95–100",
    },
    "rrt_24h": {
        "label_zh": "24小时内是否接受 RRT",
        "label_en": "Any RRT within 24 h",
        "unit": "Yes/No",
        "input_type": "binary",
        "help_en": "Whether any renal replacement therapy was used within the first 24 hours after ICU admission.",
        "default": 0,
    },
    "resp_rate_mean_24h": {
        "label_zh": "24小时平均呼吸频率",
        "label_en": "Mean respiratory rate within 24 h",
        "unit": "breaths/min",
        "input_type": "continuous",
        "help_en": "Mean respiratory rate during the first 24 hours after ICU admission.",
        "default": 18.0,
        "min": 0.0,
        "max": 80.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "12–20",
    },
    "mv_24h": {
        "label_zh": "24小时内是否机械通气",
        "label_en": "Mechanical ventilation within 24 h",
        "unit": "Yes/No",
        "input_type": "binary",
        "help_en": "Whether mechanical ventilation was used within the first 24 hours after ICU admission.",
        "default": 0,
    },
    "lactate_closest_around_icu": {
        "label_zh": "入ICU附近乳酸",
        "label_en": "Lactate closest around ICU admission",
        "unit": "mmol/L",
        "input_type": "continuous",
        "help_en": "Closest lactate value around ICU admission.",
        "default": 1.5,
        "min": 0.0,
        "max": 25.0,
        "step": 0.1,
        "format": "%.2f",
        "normal_range": "0.5–2.0",
    },
    "heart_rate_mean_24h": {
        "label_zh": "24小时平均心率",
        "label_en": "Mean heart rate within 24 h",
        "unit": "beats/min",
        "input_type": "continuous",
        "help_en": "Mean heart rate during the first 24 hours after ICU admission.",
        "default": 90.0,
        "min": 0.0,
        "max": 220.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "60–100",
    },
    "creatinine_closest_around_icu": {
        "label_zh": "入ICU附近肌酐",
        "label_en": "Creatinine closest around ICU admission",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_en": "Closest creatinine value around ICU admission.",
        "default": 1.0,
        "min": 0.0,
        "max": 20.0,
        "step": 0.01,
        "format": "%.2f",
        "normal_range": "0.6–1.3",
    },
    "bun_closest_around_icu": {
        "label_zh": "入ICU附近 BUN",
        "label_en": "BUN closest around ICU admission",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_en": "Closest blood urea nitrogen value around ICU admission.",
        "default": 15.0,
        "min": 0.0,
        "max": 150.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "7–20",
    },
    "bilirubin_total_closest_around_icu": {
        "label_zh": "入ICU附近总胆红素",
        "label_en": "Total bilirubin closest around ICU admission",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_en": "Closest total bilirubin value around ICU admission.",
        "default": 0.8,
        "min": 0.0,
        "max": 40.0,
        "step": 0.01,
        "format": "%.2f",
        "normal_range": "0.2–1.2",
    },
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def as_series_like(obj) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.Series):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.Series(obj, dtype=float)
    try:
        return pd.Series(obj, dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def merge_feature_meta(feature_cols, bundle_meta):
    meta = {}
    bundle_meta = bundle_meta or {}
    for feat in feature_cols:
        merged = dict(DEFAULT_FEATURE_META.get(feat, {}))
        override = bundle_meta.get(feat, {}) if isinstance(bundle_meta, dict) else {}
        merged.update(override)
        merged.setdefault("label_zh", feat)
        merged.setdefault("label_en", feat)
        merged.setdefault("unit", "")
        merged.setdefault("input_type", "continuous")
        merged.setdefault("default", 0.0)
        merged.setdefault("step", 0.1)
        merged.setdefault("format", "%.2f")
        meta[feat] = merged
    return meta


def load_defaults(feature_cols, medians: pd.Series, feature_meta, example_patient):
    defaults = {}
    for feat in feature_cols:
        if feat in medians.index and pd.notna(medians[feat]):
            defaults[feat] = float(medians[feat])
        else:
            defaults[feat] = float(feature_meta[feat].get("default", 0.0))
    if isinstance(example_patient, dict):
        example = {feat: float(example_patient.get(feat, defaults[feat])) for feat in feature_cols}
    else:
        example = defaults.copy()
    return defaults, example


def initialize_state(feature_cols, defaults):
    for feat in feature_cols:
        st.session_state.setdefault(f"input_{feat}", defaults[feat])


def apply_state(values):
    for feat, value in values.items():
        st.session_state[f"input_{feat}"] = value


def prepare_input_frame(feature_cols):
    data = {feat: st.session_state[f"input_{feat}"] for feat in feature_cols}
    df = pd.DataFrame([data], columns=feature_cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def locate_logistic_estimator(model):
    if hasattr(model, "named_steps"):
        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "coef_") and hasattr(step, "predict_proba"):
                return step
    if hasattr(model, "coef_") and hasattr(model, "predict_proba"):
        return model
    return None


def build_coefficient_table(bundle, model, feature_cols, feature_meta):
    exported = bundle.get("coefficient_table")
    if exported is not None:
        df = pd.DataFrame(exported).copy()
        if "feature" not in df.columns:
            return None, False
        if "coefficient" in df.columns and "odds_ratio" not in df.columns:
            df["odds_ratio"] = np.exp(df["coefficient"])
        df["label_en"] = df["feature"].map(lambda x: feature_meta.get(x, {}).get("label_en", x))
        return df, True

    estimator = locate_logistic_estimator(model)
    if estimator is None or not hasattr(estimator, "coef_"):
        return None, False

    coef = np.asarray(estimator.coef_)
    if coef.ndim != 2 or coef.shape[0] != 1 or coef.shape[1] != len(feature_cols):
        return None, False

    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "label_en": [feature_meta.get(f, {}).get("label_en", f) for f in feature_cols],
            "coefficient": coef[0],
        }
    )
    df["odds_ratio"] = np.exp(df["coefficient"])
    df["direction_en"] = np.where(df["coefficient"] >= 0, "Higher value -> higher risk", "Higher value -> lower risk")
    return df, False


def classify_risk(prob, threshold, ruleout, rulein):
    if ruleout is not None and rulein is not None and ruleout < rulein:
        if prob < ruleout:
            return "Low risk"
        if prob >= rulein:
            return "High risk"
        return "Intermediate risk"
    return "Above threshold" if prob >= threshold else "Below threshold"


if not MODEL_PATH.exists():
    st.error(APP_TEXT["missing_model_error"])
    st.stop()

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(APP_TEXT["bundle_error"])
    st.exception(exc)
    st.stop()

model = bundle.get("model")
feature_cols = list(bundle.get("feature_cols", []))
if model is None or not feature_cols:
    st.error("LR.pkl is missing the minimum required fields: model / feature_cols.")
    st.stop()

raw_classes = bundle.get("class_labels", bundle.get("classes", [APP_TEXT["binary_negative"], APP_TEXT["binary_positive"]]))
classes = [str(x) for x in raw_classes]
if len(classes) != 2:
    st.error(APP_TEXT["binary_model_error"])
    st.stop()

positive_label = str(bundle.get("positive_label", classes[1]))
threshold = float(bundle.get("threshold", 0.5))
ruleout_threshold = bundle.get("ruleout_threshold")
rulein_threshold = bundle.get("rulein_threshold")
ruleout_threshold = float(ruleout_threshold) if ruleout_threshold is not None else None
rulein_threshold = float(rulein_threshold) if rulein_threshold is not None else None

train_median = as_series_like(bundle.get("train_median"))
feature_meta = merge_feature_meta(feature_cols, bundle.get("feature_meta"))
defaults, example_patient = load_defaults(feature_cols, train_median, feature_meta, bundle.get("example_patient"))
initialize_state(feature_cols, defaults)

study_title = bundle.get("study_title_en", APP_TEXT["title"])
model_name = bundle.get("model_name", "Logistic Regression")

st.title(study_title)
st.subheader(APP_TEXT["subtitle"])
st.caption(APP_TEXT["caption"])

with st.sidebar:
    st.header(APP_TEXT["section_model"])
    st.write(f"**Model**: {model_name}")
    st.write(f"**Outcome**: {APP_TEXT['binary_positive']}")
    st.write(f"**Number of variables**: {len(feature_cols)}")
    st.write(f"**Threshold**: {threshold:.3f}")
    if ruleout_threshold is not None and rulein_threshold is not None:
        st.write(f"**Rule-out / Rule-in**: {ruleout_threshold:.3f} / {rulein_threshold:.3f}")

    st.header(APP_TEXT["section_notice"])
    st.write(APP_TEXT["notice_text"])

toolbar_col1, toolbar_col2 = st.columns(2)
if toolbar_col1.button(APP_TEXT["fill_default"], use_container_width=True):
    apply_state(defaults)
if toolbar_col2.button(APP_TEXT["fill_example"], use_container_width=True):
    apply_state(example_patient)

st.markdown("---")
st.header(APP_TEXT["section_input"])
st.caption("Use verified patient-specific values whenever possible. Initial values are provided only for demonstration.")

with st.form("prediction_form"):
    input_cols = st.columns(2)
    for idx, feat in enumerate(feature_cols):
        meta = feature_meta[feat]
        label = meta.get("label_en", feat)
        unit = meta.get("unit", "")
        label_show = f"{label} ({unit})" if unit else label
        help_text = meta.get("help_en", "")
        if meta.get("normal_range"):
            help_text = f"{help_text}\nReference range: {meta['normal_range']}"
        help_text = f"{help_text}\n{APP_TEXT['input_help_prefix']}: {feat}".strip()
        state_key = f"input_{feat}"
        with input_cols[idx % 2]:
            if meta.get("input_type") == "binary":
                current_value = int(round(float(st.session_state[state_key])))
                st.selectbox(
                    label_show,
                    options=[0, 1],
                    index=1 if current_value == 1 else 0,
                    format_func=lambda x: APP_TEXT["yes"] if x == 1 else APP_TEXT["no"],
                    key=state_key,
                    help=help_text,
                )
            else:
                st.number_input(
                    label_show,
                    min_value=float(meta.get("min", -1e9)),
                    max_value=float(meta.get("max", 1e9)),
                    step=float(meta.get("step", 0.1)),
                    format=str(meta.get("format", "%.2f")),
                    key=state_key,
                    help=help_text,
                )
    submitted = st.form_submit_button(APP_TEXT["predict_button"], use_container_width=True)

if submitted:
    x_row = prepare_input_frame(feature_cols)
    if x_row.isna().any().any():
        st.error("Missing or non-numeric input was detected. Please review the entered values.")
        st.stop()

    try:
        proba = np.asarray(model.predict_proba(x_row)[0], dtype=float)
    except Exception as exc:
        st.error("Model inference failed. Please verify that LR.pkl matches the exact feature order used in the app.")
        st.exception(exc)
        st.stop()

    if proba.shape[0] != 2:
        st.error(APP_TEXT["binary_model_error"])
        st.stop()

    class_to_proba = dict(zip(classes, proba))
    if positive_label not in class_to_proba:
        positive_label = classes[1]
    unfavorable_prob = float(class_to_proba[positive_label])
    predicted_label = positive_label if unfavorable_prob >= threshold else [c for c in classes if c != positive_label][0]
    risk_band = classify_risk(unfavorable_prob, threshold, ruleout_threshold, rulein_threshold)

    st.markdown("---")
    st.header(APP_TEXT["section_result"])
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric(APP_TEXT["risk_probability"], f"{unfavorable_prob:.3f}")
    result_col2.metric(APP_TEXT["predicted_label"], predicted_label)
    result_col3.metric(APP_TEXT["risk_band"], risk_band)
    st.caption(f"{APP_TEXT['threshold_label']}: {threshold:.3f}")

    st.header(APP_TEXT["section_prob"])
    prob_df = pd.DataFrame(
        {
            APP_TEXT["probability_table_class"]: classes,
            APP_TEXT["probability_table_probability"]: [float(class_to_proba[c]) for c in classes],
        }
    ).sort_values(APP_TEXT["probability_table_probability"], ascending=False)
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    summary_df = pd.DataFrame(
        [
            {"item": "Predicted label", "value": predicted_label},
            {"item": "Unfavorable probability", "value": round(unfavorable_prob, 6)},
            {"item": "Threshold", "value": round(threshold, 6)},
            {"item": "Risk band", "value": risk_band},
        ]
    )
    st.download_button(
        APP_TEXT["download"],
        data=summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lr_prediction_summary.csv",
        mime="text/csv",
    )

    st.header(APP_TEXT["section_interpret"])
    coef_table, exported_coef = build_coefficient_table(bundle, model, feature_cols, feature_meta)
    if coef_table is None:
        st.info(APP_TEXT["no_interpret_note"])
    else:
        st.caption(APP_TEXT["coef_exported_note"] if exported_coef else APP_TEXT["coef_fallback_note"])
        plot_df = coef_table.copy()
        plot_df["abs_coef"] = np.abs(plot_df["coefficient"])
        plot_df = plot_df.sort_values("abs_coef", ascending=True).tail(min(10, len(plot_df)))

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#c0392b" if v >= 0 else "#2980b9" for v in plot_df["coefficient"]]
        ax.barh(plot_df.get("label_en", plot_df["feature"]), plot_df["coefficient"], color=colors)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_title("Direction and relative magnitude of LR coefficients")
        ax.set_xlabel("Coefficient")
        plt.tight_layout()
        st.pyplot(fig)

        show_cols = [c for c in ["label_en", "feature", "coefficient", "odds_ratio", "direction_en"] if c in coef_table.columns]
        st.dataframe(coef_table[show_cols], use_container_width=True, hide_index=True)

    st.header(APP_TEXT["section_snapshot"])
    snapshot_rows = []
    for feat in feature_cols:
        meta = feature_meta[feat]
        raw_value = st.session_state[f"input_{feat}"]
        if meta.get("input_type") == "binary":
            display_value = APP_TEXT["yes"] if int(raw_value) == 1 else APP_TEXT["no"]
        else:
            display_value = raw_value
        snapshot_rows.append(
            {
                APP_TEXT["snapshot_feature"]: meta.get("label_en", feat),
                APP_TEXT["snapshot_value"]: display_value,
                APP_TEXT["snapshot_unit"]: meta.get("unit", ""),
            }
        )
    st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True, hide_index=True)
