# -*- coding: utf-8 -*-
"""
Deployment-ready Streamlit app for the final AP ICU LR model.

Expected files in the same directory:
    - app.py
    - LR.pkl
    - requirements.txt

The LR.pkl bundle should ideally contain:
    model, feature_cols, class_labels, positive_label, threshold,
    ruleout_threshold, rulein_threshold, train_median, feature_meta,
    example_patient, coefficient_table, study_title_en, model_name
"""

from pathlib import Path
import traceback

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AP ICU Trajectory Risk Predictor",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "LR.pkl"

APP_TEXT = {
    "title": "AP ICU Trajectory Risk Predictor",
    "subtitle": "Research companion web tool based on the final Logistic Regression model",
    "caption": "This tool is for research demonstration only and not for clinical decision-making.",
    "model_file": "Current model file: LR.pkl",
    "model_section": "Model overview",
    "input_section": "Patient inputs",
    "input_caption": "Please enter verified patient data when available. The initial values are for demonstration only.",
    "result_section": "Prediction results",
    "prob_section": "Probability output",
    "interpret_section": "Coefficient interpretation",
    "snapshot_section": "Input summary",
    "notice_section": "Notes and disclaimer",
    "notice_text": (
        "1. This tool should only be used for research communication and manuscript support.\n"
        "2. It does not replace clinical judgement.\n"
        "3. Please ensure that LR.pkl exactly matches the final manuscript model.\n"
        "4. If any input appears implausible, please verify the underlying source data."
    ),
    "reset_button": "Reset to training defaults",
    "example_button": "Load example patient",
    "predict_button": "Run prediction",
    "outcome_positive": "Unfavorable",
    "outcome_negative": "Favorable",
    "predicted_label": "Predicted class",
    "risk_probability": "Probability of unfavorable trajectory",
    "risk_band": "Risk category",
    "threshold_label": "Decision threshold",
    "download": "Download result summary CSV",
    "binary_yes": "Yes",
    "binary_no": "No",
    "prob_table_class": "Class",
    "prob_table_probability": "Probability",
    "snapshot_feature": "Feature",
    "snapshot_value": "Value",
    "snapshot_unit": "Unit",
    "coef_exported_note": "The table below uses the coefficient summary exported from the training workflow.",
    "coef_fallback_note": (
        "The coefficients below were extracted directly from the loaded LR model. "
        "If preprocessing includes scaling or transformations, they should be interpreted primarily as direction and relative weight."
    ),
    "coef_missing_note": "Coefficient interpretation is unavailable for the current model bundle.",
    "missing_model_error": "LR.pkl was not found. Please place the final model bundle in the same folder as app.py.",
    "bundle_error": "Failed to load LR.pkl. Please verify that the file exists, is not corrupted, and contains the required fields.",
    "bundle_field_error": "The model bundle is missing required fields: model or feature_cols.",
    "binary_model_error": "This web app is intended for the final binary LR model only.",
    "predict_error": "Prediction failed. Please verify that LR.pkl matches the current feature order and preprocessing logic.",
    "input_error": "At least one input is missing or invalid. Please review all fields.",
    "debug_section": "Debug details",
    "compat_notice": "Legacy LogisticRegression compatibility patch applied successfully.",
}

DEFAULT_FEATURE_META = {
    "vaso_any_24h": {
        "label_en": "Any vasopressor within 24 h",
        "unit": "",
        "input_type": "binary",
        "help_en": "Whether any vasopressor was used within the first 24 hours after ICU admission.",
        "default": 0,
    },
    "spo2_mean_24h": {
        "label_en": "Mean SpO2 within 24 h (%)",
        "unit": "%",
        "input_type": "continuous",
        "help_en": "Mean oxygen saturation during the first 24 hours after ICU admission.",
        "default": 96.0,
        "min": 50.0,
        "max": 100.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "95-100%",
    },
    "rrt_24h": {
        "label_en": "RRT within 24 h",
        "unit": "",
        "input_type": "binary",
        "help_en": "Whether any renal replacement therapy was used within the first 24 hours after ICU admission.",
        "default": 0,
    },
    "resp_rate_mean_24h": {
        "label_en": "Mean respiratory rate within 24 h (breaths/min)",
        "unit": "breaths/min",
        "input_type": "continuous",
        "help_en": "Mean respiratory rate during the first 24 hours after ICU admission.",
        "default": 18.0,
        "min": 0.0,
        "max": 80.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "12-20 breaths/min",
    },
    "mv_24h": {
        "label_en": "Mechanical ventilation within 24 h",
        "unit": "",
        "input_type": "binary",
        "help_en": "Whether mechanical ventilation was used within the first 24 hours after ICU admission.",
        "default": 0,
    },
    "lactate_closest_around_icu": {
        "label_en": "Lactate around ICU admission (mmol/L)",
        "unit": "mmol/L",
        "input_type": "continuous",
        "help_en": "Closest lactate value around ICU admission.",
        "default": 1.5,
        "min": 0.0,
        "max": 25.0,
        "step": 0.1,
        "format": "%.2f",
        "normal_range": "0.5-2.0 mmol/L",
    },
    "heart_rate_mean_24h": {
        "label_en": "Mean heart rate within 24 h (beats/min)",
        "unit": "beats/min",
        "input_type": "continuous",
        "help_en": "Mean heart rate during the first 24 hours after ICU admission.",
        "default": 90.0,
        "min": 0.0,
        "max": 220.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "60-100 beats/min",
    },
    "creatinine_closest_around_icu": {
        "label_en": "Creatinine around ICU admission (mg/dL)",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_en": "Closest creatinine value around ICU admission.",
        "default": 1.0,
        "min": 0.0,
        "max": 20.0,
        "step": 0.01,
        "format": "%.2f",
        "normal_range": "0.6-1.3 mg/dL",
    },
    "bun_closest_around_icu": {
        "label_en": "Blood urea nitrogen around ICU admission (mg/dL)",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_en": "Closest blood urea nitrogen value around ICU admission.",
        "default": 15.0,
        "min": 0.0,
        "max": 150.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "7-20 mg/dL",
    },
    "bilirubin_total_closest_around_icu": {
        "label_en": "Total bilirubin around ICU admission (mg/dL)",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_en": "Closest total bilirubin value around ICU admission.",
        "default": 0.8,
        "min": 0.0,
        "max": 40.0,
        "step": 0.01,
        "format": "%.2f",
        "normal_range": "0.2-1.2 mg/dL",
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
        merged.setdefault("label_en", DEFAULT_FEATURE_META.get(feat, {}).get("label_en", feat))
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
        example = {}
        for feat in feature_cols:
            raw_value = example_patient.get(feat, defaults[feat])
            if pd.isna(raw_value):
                raw_value = defaults[feat]
            example[feat] = float(raw_value)
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


def iter_model_components(model):
    yield "model", model
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            yield f"named_steps.{name}", step
    if hasattr(model, "steps"):
        for name, step in model.steps:
            yield f"steps.{name}", step
    if hasattr(model, "best_estimator_"):
        yield "best_estimator_", model.best_estimator_


def patch_logistic_regression_compat(model, feature_count):
    patch_notes = []
    seen = set()

    for location, obj in iter_model_components(model):
        if id(obj) in seen:
            continue
        seen.add(id(obj))

        if obj.__class__.__name__ != "LogisticRegression":
            continue

        if not hasattr(obj, "multi_class"):
            inferred = "ovr"
            if hasattr(obj, "classes_"):
                try:
                    inferred = "ovr" if len(obj.classes_) <= 2 else "auto"
                except Exception:
                    inferred = "ovr"
            obj.multi_class = inferred
            patch_notes.append(f"{location}: added missing multi_class={inferred!r}")

        if not hasattr(obj, "n_features_in_") and hasattr(obj, "coef_"):
            try:
                obj.n_features_in_ = int(obj.coef_.shape[1])
                patch_notes.append(f"{location}: added missing n_features_in_={obj.n_features_in_}")
            except Exception:
                if feature_count is not None:
                    obj.n_features_in_ = int(feature_count)
                    patch_notes.append(f"{location}: added fallback n_features_in_={obj.n_features_in_}")

        if not hasattr(obj, "n_iter_") and hasattr(obj, "coef_"):
            obj.n_iter_ = np.array([1], dtype=np.int32)
            patch_notes.append(f"{location}: added missing n_iter_={obj.n_iter_!r}")

        if not hasattr(obj, "classes_") and hasattr(obj, "coef_"):
            if obj.coef_.shape[0] == 1:
                obj.classes_ = np.array([0, 1])
            else:
                obj.classes_ = np.arange(obj.coef_.shape[0])
            patch_notes.append(f"{location}: added fallback classes_={obj.classes_!r}")

        if not hasattr(obj, "feature_names_in_") and feature_count is not None:
            try:
                obj.feature_names_in_ = np.array([f"feature_{i}" for i in range(int(feature_count))], dtype=object)
                patch_notes.append(f"{location}: added fallback feature_names_in_")
            except Exception:
                pass

    return patch_notes


def locate_logistic_estimator(model):
    if hasattr(model, "named_steps"):
        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "coef_") and hasattr(step, "predict_proba"):
                return step
    if hasattr(model, "coef_") and hasattr(model, "predict_proba"):
        return model
    return None


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def safe_predict_binary(model, x_row, class_labels, positive_label):
    debug_notes = []
    debug_trace = None

    patch_notes = patch_logistic_regression_compat(model, feature_count=x_row.shape[1])
    debug_notes.extend(patch_notes)

    try:
        proba = np.asarray(model.predict_proba(x_row)[0], dtype=float)
        return proba, debug_notes, debug_trace
    except Exception:
        debug_trace = traceback.format_exc()
        debug_notes.append("predict_proba() failed; attempting decision_function() fallback.")

    try:
        scores = np.asarray(model.decision_function(x_row), dtype=float)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if scores.ndim == 1:
            pos_prob = float(sigmoid(scores[0]))
            if len(class_labels) != 2:
                raise RuntimeError("Binary fallback requires exactly two class labels.")
            if positive_label == class_labels[0]:
                proba = np.array([pos_prob, 1.0 - pos_prob], dtype=float)
            else:
                proba = np.array([1.0 - pos_prob, pos_prob], dtype=float)
        else:
            proba = softmax(scores)[0]
        return proba, debug_notes, debug_trace
    except Exception:
        debug_notes.append("decision_function() fallback also failed.")
        debug_trace = (debug_trace or "") + "\n\nFallback traceback:\n" + traceback.format_exc()
        raise RuntimeError("Prediction failed for the current LR bundle.") from None


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


def render_debug(trace_text, debug_notes):
    with st.expander(APP_TEXT["debug_section"]):
        if debug_notes:
            st.write("Compatibility and runtime notes:")
            for note in debug_notes:
                st.write(f"- {note}")
        if trace_text:
            st.code(trace_text, language="text")


if not MODEL_PATH.exists():
    st.error(APP_TEXT["missing_model_error"])
    st.stop()

try:
    bundle = load_bundle()
except Exception:
    st.error(APP_TEXT["bundle_error"])
    render_debug(traceback.format_exc(), [])
    st.stop()

model = bundle.get("model")
feature_cols = list(bundle.get("feature_cols", []))
if model is None or not feature_cols:
    st.error(APP_TEXT["bundle_field_error"])
    st.stop()

raw_classes = bundle.get("class_labels", [APP_TEXT["outcome_negative"], APP_TEXT["outcome_positive"]])
class_labels = [str(x) for x in raw_classes]
if len(class_labels) != 2:
    st.error(APP_TEXT["binary_model_error"])
    st.stop()

positive_label = str(bundle.get("positive_label", class_labels[1]))
threshold = float(bundle.get("threshold", 0.5))
ruleout_threshold = bundle.get("ruleout_threshold")
rulein_threshold = bundle.get("rulein_threshold")
ruleout_threshold = float(ruleout_threshold) if ruleout_threshold is not None else None
rulein_threshold = float(rulein_threshold) if rulein_threshold is not None else None

train_median = as_series_like(bundle.get("train_median"))
feature_meta = merge_feature_meta(feature_cols, bundle.get("feature_meta"))
defaults, example_patient = load_defaults(feature_cols, train_median, feature_meta, bundle.get("example_patient"))
initialize_state(feature_cols, defaults)

compat_notes = patch_logistic_regression_compat(model, feature_count=len(feature_cols))

study_title = bundle.get("study_title_en", APP_TEXT["title"])
model_name = bundle.get("model_name", "Logistic Regression")

st.title(study_title)
st.subheader(APP_TEXT["subtitle"])
st.caption(APP_TEXT["caption"])
st.caption(APP_TEXT["model_file"])

with st.sidebar:
    st.header(APP_TEXT["model_section"])
    st.write(f"**Model name:** {model_name}")
    st.write(f"**Outcome:** {APP_TEXT['outcome_positive']}")
    st.write(f"**Number of predictors:** {len(feature_cols)}")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    if ruleout_threshold is not None and rulein_threshold is not None:
        st.write(f"**Rule-out / Rule-in:** {ruleout_threshold:.3f} / {rulein_threshold:.3f}")
    if compat_notes:
        st.info(APP_TEXT["compat_notice"])

    st.header(APP_TEXT["notice_section"])
    st.write(APP_TEXT["notice_text"])

toolbar_col1, toolbar_col2 = st.columns(2)
if toolbar_col1.button(APP_TEXT["reset_button"], use_container_width=True):
    apply_state(defaults)
if toolbar_col2.button(APP_TEXT["example_button"], use_container_width=True):
    apply_state(example_patient)

st.markdown("---")
st.header(APP_TEXT["input_section"])
st.caption(APP_TEXT["input_caption"])

with st.form("prediction_form"):
    input_cols = st.columns(2)
    for idx, feat in enumerate(feature_cols):
        meta = feature_meta[feat]
        label = meta.get("label_en", feat)
        help_text = meta.get("help_en", "")
        if meta.get("normal_range"):
            help_text = f"{help_text}\nReference range: {meta['normal_range']}"
        help_text = f"{help_text}\nVariable code: {feat}".strip()
        state_key = f"input_{feat}"

        with input_cols[idx % 2]:
            if meta.get("input_type") == "binary":
                current_value = int(round(float(st.session_state[state_key])))
                st.selectbox(
                    label,
                    options=[0, 1],
                    index=1 if current_value == 1 else 0,
                    format_func=lambda x: APP_TEXT["binary_yes"] if x == 1 else APP_TEXT["binary_no"],
                    key=state_key,
                    help=help_text,
                )
            else:
                st.number_input(
                    label,
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
        st.error(APP_TEXT["input_error"])
        st.stop()

    try:
        proba, predict_notes, predict_trace = safe_predict_binary(model, x_row, class_labels, positive_label)
    except Exception:
        st.error(APP_TEXT["predict_error"])
        render_debug(traceback.format_exc(), compat_notes)
        st.stop()

    if proba.shape[0] != 2:
        st.error(APP_TEXT["binary_model_error"])
        render_debug(predict_trace, compat_notes + predict_notes)
        st.stop()

    class_to_proba = dict(zip(class_labels, proba))
    if positive_label not in class_to_proba:
        positive_label = class_labels[1]

    unfavorable_prob = float(class_to_proba[positive_label])
    predicted_label = positive_label if unfavorable_prob >= threshold else [c for c in class_labels if c != positive_label][0]
    risk_band = classify_risk(unfavorable_prob, threshold, ruleout_threshold, rulein_threshold)

    st.markdown("---")
    st.header(APP_TEXT["result_section"])
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric(APP_TEXT["risk_probability"], f"{unfavorable_prob:.3f}")
    result_col2.metric(APP_TEXT["predicted_label"], predicted_label)
    result_col3.metric(APP_TEXT["risk_band"], risk_band)
    st.caption(f"{APP_TEXT['threshold_label']}: {threshold:.3f}")

    st.header(APP_TEXT["prob_section"])
    prob_df = pd.DataFrame(
        {
            APP_TEXT["prob_table_class"]: class_labels,
            APP_TEXT["prob_table_probability"]: [float(class_to_proba[c]) for c in class_labels],
        }
    ).sort_values(APP_TEXT["prob_table_probability"], ascending=False)
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    summary_df = pd.DataFrame(
        [
            {"item": "Predicted class", "value": predicted_label},
            {"item": "Probability of unfavorable trajectory", "value": round(unfavorable_prob, 6)},
            {"item": "Decision threshold", "value": round(threshold, 6)},
            {"item": "Risk category", "value": risk_band},
        ]
    )
    st.download_button(
        APP_TEXT["download"],
        data=summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lr_prediction_summary.csv",
        mime="text/csv",
    )

    st.header(APP_TEXT["interpret_section"])
    coef_table, exported_coef = build_coefficient_table(bundle, model, feature_cols, feature_meta)
    if coef_table is None:
        st.info(APP_TEXT["coef_missing_note"])
    else:
        st.caption(APP_TEXT["coef_exported_note"] if exported_coef else APP_TEXT["coef_fallback_note"])
        plot_df = coef_table.copy()
        plot_df["abs_coef"] = np.abs(plot_df["coefficient"])
        plot_df = plot_df.sort_values("abs_coef", ascending=True).tail(min(10, len(plot_df)))

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#c0392b" if value >= 0 else "#2980b9" for value in plot_df["coefficient"]]
        ax.barh(plot_df.get("label_en", plot_df["feature"]), plot_df["coefficient"], color=colors)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_title("Coefficient direction and relative magnitude")
        ax.set_xlabel("Coefficient")
        plt.tight_layout()
        st.pyplot(fig)

        show_cols = [c for c in ["label_en", "feature", "coefficient", "odds_ratio", "direction_en"] if c in coef_table.columns]
        st.dataframe(coef_table[show_cols], use_container_width=True, hide_index=True)

    st.header(APP_TEXT["snapshot_section"])
    snapshot_rows = []
    for feat in feature_cols:
        meta = feature_meta[feat]
        raw_value = st.session_state[f"input_{feat}"]
        if meta.get("input_type") == "binary":
            display_value = APP_TEXT["binary_yes"] if int(raw_value) == 1 else APP_TEXT["binary_no"]
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

    if compat_notes or predict_notes or predict_trace:
        render_debug(predict_trace, compat_notes + predict_notes)
