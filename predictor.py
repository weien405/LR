# -*- coding: utf-8 -*-
"""
AP ICU 预后预测工具

说明：
1. 本页面用于加载最终 LR.pkl 模型并进行推理；
2. 页面仅用于研究展示和论文配套说明，不可替代临床判断；
3. 模型 bundle 至少需要包含：
   - model
   - feature_cols
4. 推荐包含：
   - class_labels
   - positive_label
   - threshold
   - ruleout_threshold
   - rulein_threshold
   - train_median
   - feature_meta
   - example_patient
   - coefficient_table
   - study_title_zh
   - model_name
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AP ICU 预后预测工具",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "LR.pkl"

APP_TEXT = {
    "title": "AP ICU 预后预测工具",
    "subtitle": "基于最终 Logistic Regression（LR）模型的研究配套网页工具",
    "caption": "本工具仅用于研究展示和论文配套说明，不可替代临床判断。请确保加载的 LR.pkl 与论文最终模型完全一致。",
    "model_file": "当前模型文件：LR.pkl",
    "model_section": "模型说明",
    "input_section": "患者输入",
    "input_caption": "建议使用经过核对的患者真实数据；初始值仅用于页面演示。",
    "result_section": "预测结果",
    "prob_section": "概率输出",
    "interpret_section": "模型解释",
    "snapshot_section": "当前输入摘要",
    "notice_section": "使用说明与免责声明",
    "notice_text": (
        "1. 请按论文最终定义输入前24小时或入 ICU 附近变量；"
        "2. 页面默认值仅用于演示，不可替代真实患者数据；"
        "3. 若输入值明显超出生理合理范围，请先核对数据来源；"
        "4. 本工具仅用于研究交流、补充材料和在线演示，不作为独立临床决策依据。"
    ),
    "reset_button": "恢复训练集默认值",
    "example_button": "填入示例患者",
    "predict_button": "开始预测",
    "outcome_positive": "不良轨迹",
    "outcome_negative": "非不良轨迹",
    "predicted_label": "预测分类",
    "risk_probability": "不良轨迹概率",
    "risk_band": "风险分层",
    "threshold_label": "预设判定阈值",
    "download": "下载结果摘要 CSV",
    "binary_yes": "是",
    "binary_no": "否",
    "prob_table_class": "类别",
    "prob_table_probability": "概率",
    "snapshot_feature": "变量",
    "snapshot_value": "输入值",
    "snapshot_unit": "单位",
    "coef_exported_note": "以下为训练脚本导出的正式变量解释摘要，适合论文配套展示。",
    "coef_fallback_note": "以下系数由当前 LR 模型直接提取。若训练流程包含标准化或其他变换，这些系数更适合用于方向解释，而不宜直接当作原始量纲效应。",
    "coef_missing_note": "当前 bundle 未提供可展示的系数解释信息，页面仅显示概率输出和输入摘要。",
    "missing_model_error": "未检测到 LR.pkl。请将最终 LR 模型文件放在 predictor.py 同目录下后重新部署。",
    "bundle_error": "无法加载 LR.pkl。请确认模型文件存在、未损坏，且 bundle 字段完整。",
    "bundle_field_error": "LR.pkl 缺少必要字段：model 或 feature_cols。",
    "binary_model_error": "当前网页仅适用于二分类 LR 模型，但加载对象并非二分类模型。",
    "predict_error": "模型推理失败。请检查 LR.pkl 是否与当前变量顺序完全一致。",
    "input_error": "检测到缺失或非数值输入，请重新检查。",
}

DEFAULT_FEATURE_META = {
    "vaso_any_24h": {
        "label_zh": "24小时内是否使用升压药",
        "unit": "",
        "input_type": "binary",
        "help_zh": "ICU 入科后前24小时内是否曾使用任一升压药。",
        "default": 0,
    },
    "spo2_mean_24h": {
        "label_zh": "24小时内平均SpO₂（%）",
        "unit": "%",
        "input_type": "continuous",
        "help_zh": "ICU 入科后前24小时内平均血氧饱和度。",
        "default": 96.0,
        "min": 50.0,
        "max": 100.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "95–100%",
    },
    "rrt_24h": {
        "label_zh": "24小时内是否接受RRT",
        "unit": "",
        "input_type": "binary",
        "help_zh": "ICU 入科后前24小时内是否接受任一肾脏替代治疗。",
        "default": 0,
    },
    "resp_rate_mean_24h": {
        "label_zh": "24小时内平均呼吸频率（次/分）",
        "unit": "次/分",
        "input_type": "continuous",
        "help_zh": "ICU 入科后前24小时内平均呼吸频率。",
        "default": 18.0,
        "min": 0.0,
        "max": 80.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "12–20 次/分",
    },
    "mv_24h": {
        "label_zh": "24小时内是否机械通气",
        "unit": "",
        "input_type": "binary",
        "help_zh": "ICU 入科后前24小时内是否接受机械通气。",
        "default": 0,
    },
    "lactate_closest_around_icu": {
        "label_zh": "ICU入科附近乳酸（mmol/L）",
        "unit": "mmol/L",
        "input_type": "continuous",
        "help_zh": "入 ICU 前后最近一次乳酸检测值。",
        "default": 1.5,
        "min": 0.0,
        "max": 25.0,
        "step": 0.1,
        "format": "%.2f",
        "normal_range": "0.5–2.0 mmol/L",
    },
    "heart_rate_mean_24h": {
        "label_zh": "24小时内平均心率（次/分）",
        "unit": "次/分",
        "input_type": "continuous",
        "help_zh": "ICU 入科后前24小时内平均心率。",
        "default": 90.0,
        "min": 0.0,
        "max": 220.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "60–100 次/分",
    },
    "creatinine_closest_around_icu": {
        "label_zh": "ICU入科附近肌酐（mg/dL）",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_zh": "入 ICU 前后最近一次肌酐值。",
        "default": 1.0,
        "min": 0.0,
        "max": 20.0,
        "step": 0.01,
        "format": "%.2f",
        "normal_range": "0.6–1.3 mg/dL",
    },
    "bun_closest_around_icu": {
        "label_zh": "ICU入科附近尿素氮（BUN, mg/dL）",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_zh": "入 ICU 前后最近一次尿素氮值。",
        "default": 15.0,
        "min": 0.0,
        "max": 150.0,
        "step": 0.1,
        "format": "%.1f",
        "normal_range": "7–20 mg/dL",
    },
    "bilirubin_total_closest_around_icu": {
        "label_zh": "ICU入科附近总胆红素（mg/dL）",
        "unit": "mg/dL",
        "input_type": "continuous",
        "help_zh": "入 ICU 前后最近一次总胆红素值。",
        "default": 0.8,
        "min": 0.0,
        "max": 40.0,
        "step": 0.01,
        "format": "%.2f",
        "normal_range": "0.2–1.2 mg/dL",
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
        df["label_zh"] = df["feature"].map(lambda x: feature_meta.get(x, {}).get("label_zh", x))
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
            "label_zh": [feature_meta.get(f, {}).get("label_zh", f) for f in feature_cols],
            "coefficient": coef[0],
        }
    )
    df["odds_ratio"] = np.exp(df["coefficient"])
    df["direction_zh"] = np.where(df["coefficient"] >= 0, "值越高，风险越高", "值越高，风险越低")
    return df, False


def classify_risk(prob, threshold, ruleout, rulein):
    if ruleout is not None and rulein is not None and ruleout < rulein:
        if prob < ruleout:
            return "低风险"
        if prob >= rulein:
            return "高风险"
        return "中间风险"
    return "高于阈值" if prob >= threshold else "低于阈值"


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
    st.error(APP_TEXT["bundle_field_error"])
    st.stop()

raw_classes = bundle.get("class_labels", [APP_TEXT["outcome_negative"], APP_TEXT["outcome_positive"]])
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

study_title = bundle.get("study_title_zh", APP_TEXT["title"])
model_name = bundle.get("model_name", "Logistic Regression")

st.title(study_title)
st.subheader(APP_TEXT["subtitle"])
st.caption(APP_TEXT["caption"])
st.caption(APP_TEXT["model_file"])

with st.sidebar:
    st.header(APP_TEXT["model_section"])
    st.write(f"**模型名称：** {model_name}")
    st.write(f"**预测结局：** {APP_TEXT['outcome_positive']}")
    st.write(f"**变量数量：** {len(feature_cols)}")
    st.write(f"**判定阈值：** {threshold:.3f}")
    if ruleout_threshold is not None and rulein_threshold is not None:
        st.write(f"**Rule-out / Rule-in：** {ruleout_threshold:.3f} / {rulein_threshold:.3f}")

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
        label = meta.get("label_zh", feat)
        help_text = meta.get("help_zh", "")
        if meta.get("normal_range"):
            help_text = f"{help_text}\n参考范围：{meta['normal_range']}"
        help_text = f"{help_text}\n变量代码：{feat}".strip()
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
        proba = np.asarray(model.predict_proba(x_row)[0], dtype=float)
    except Exception as exc:
        st.error(APP_TEXT["predict_error"])
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
    st.header(APP_TEXT["result_section"])
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric(APP_TEXT["risk_probability"], f"{unfavorable_prob:.3f}")
    result_col2.metric(APP_TEXT["predicted_label"], predicted_label)
    result_col3.metric(APP_TEXT["risk_band"], risk_band)
    st.caption(f"{APP_TEXT['threshold_label']}：{threshold:.3f}")

    st.header(APP_TEXT["prob_section"])
    prob_df = pd.DataFrame(
        {
            APP_TEXT["prob_table_class"]: classes,
            APP_TEXT["prob_table_probability"]: [float(class_to_proba[c]) for c in classes],
        }
    ).sort_values(APP_TEXT["prob_table_probability"], ascending=False)
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
        ax.barh(plot_df.get("label_zh", plot_df["feature"]), plot_df["coefficient"], color=colors)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_title("变量系数方向与相对强度")
        ax.set_xlabel("Coefficient")
        plt.tight_layout()
        st.pyplot(fig)

        show_cols = [c for c in ["label_zh", "feature", "coefficient", "odds_ratio", "direction_zh"] if c in coef_table.columns]
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
                APP_TEXT["snapshot_feature"]: meta.get("label_zh", feat),
                APP_TEXT["snapshot_value"]: display_value,
                APP_TEXT["snapshot_unit"]: meta.get("unit", ""),
            }
        )
    st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True, hide_index=True)
