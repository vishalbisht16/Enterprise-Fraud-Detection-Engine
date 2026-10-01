
import logging
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TRANSACTION_TYPES = ("DEBIT", "TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN")
RECEIVER_BALANCE_TYPES = ("TRANSFER", "PAYMENT")


def validate_transaction_inputs(
    tx_type,
    amount,
    old_orig,
    new_orig,
    old_dest=None,
    new_dest=None,
):
    if tx_type not in TRANSACTION_TYPES:
        raise ValueError("Select a supported transaction type.")

    values = {
        "Transaction amount": amount,
        "Sender old balance": old_orig,
        "Sender new balance": new_orig,
    }
    if tx_type in RECEIVER_BALANCE_TYPES:
        values["Receiver old balance"] = old_dest
        values["Receiver new balance"] = new_dest

    for label, value in values.items():
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{label} must be a finite number.")
        if float(value) < 0:
            raise ValueError(f"{label} cannot be negative.")

    if float(amount) <= 0:
        raise ValueError("Transaction amount must be greater than zero.")


def ledger_mismatch_reasons(tx_type, amount, old_orig, new_orig, old_dest=0.0, new_dest=0.0):
    reasons = []
    expected_new_orig = old_orig + amount if tx_type == "CASH_IN" else old_orig - amount
    if abs(new_orig - expected_new_orig) >= 1.00:
        label = "CASH_IN" if tx_type == "CASH_IN" else "Sender"
        reasons.append(
            f"{label} Balance Mismatch: Expected ${expected_new_orig:,.2f}, found ${new_orig:,.2f}"
        )

    if tx_type in RECEIVER_BALANCE_TYPES:
        expected_new_dest = old_dest + amount
        if abs(new_dest - expected_new_dest) >= 1.00:
            reasons.append(
                f"Receiver Balance Mismatch: Expected ${expected_new_dest:,.2f}, found ${new_dest:,.2f}"
            )
    return reasons


def build_model_features(tx_type, amount, old_orig, new_orig, old_dest=0.0, new_dest=0.0):
    has_receiver = tx_type in RECEIVER_BALANCE_TYPES
    model_old_dest = old_dest if has_receiver else 0.0
    model_new_dest = new_dest if has_receiver else 0.0
    error_orig = (
        new_orig - amount - old_orig
        if tx_type == "CASH_IN"
        else new_orig + amount - old_orig
    )
    error_dest = model_old_dest + amount - model_new_dest if has_receiver else 0.0

    return {
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": model_old_dest,
        "newbalanceDest": model_new_dest,
        "errorBalanceOrig": error_orig,
        "errorBalanceDest": error_dest,
        f"type_{tx_type}": 1,
    }


st.set_page_config(page_title="Enterprise Fraud Engine", layout="wide")

@st.cache_resource
def load_pipeline():
    artifact_path = Path(__file__).resolve().parent.parent / 'models' / 'final_fraud_pipeline.joblib'
    return joblib.load(artifact_path)

artifact = load_pipeline()
model = artifact['model']
feature_columns = artifact['feature_columns']
threshold = artifact['optimal_threshold']
model_version = artifact.get('model_version', 'unversioned')

st.session_state.setdefault('evaluation_count', 0)
st.session_state.setdefault('high_risk_count', 0)
st.session_state.setdefault('rule_alert_count', 0)

st.title("🛡️ Enterprise Fraud Detection Engine")
st.caption("Production ML System with Dual-Side Core Banking Ledger Rules")

# ---------------------------------------------------------
# REAL-TIME TRANSACTION INSPECTOR
# ---------------------------------------------------------
st.subheader("Real-Time Transaction Input")
c1, c2, c3 = st.columns(3)
with c1:
    tx_type = st.selectbox("Transaction Type", ["DEBIT", "TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"])
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=5000.0)
with c2:
    old_orig = st.number_input("Sender Old Balance", min_value=0.0, value=10000.0)
    new_orig = st.number_input("Sender New Balance", min_value=0.0, value=15000.0 if tx_type == "CASH_IN" else 5000.0)
with c3:
    if tx_type in RECEIVER_BALANCE_TYPES:
        old_dest = st.number_input("Receiver Old Balance", min_value=0.0, value=1000.0)
        new_dest = st.number_input("Receiver New Balance", min_value=0.0, value=6000.0)
    else:
        st.info("Receiver balance is not applicable for this transaction type.")
        old_dest = 0.0
        new_dest = 0.0

if st.button("Evaluate Transaction Risk", type="primary"):
    try:
        validate_transaction_inputs(tx_type, amount, old_orig, new_orig, old_dest, new_dest)
    except (TypeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    rule_reasons = ledger_mismatch_reasons(tx_type, amount, old_orig, new_orig, old_dest, new_dest)
    rule_triggered = len(rule_reasons) > 0
    input_data = pd.DataFrame([build_model_features(
        tx_type, amount, old_orig, new_orig, old_dest, new_dest
    )])

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]

    ml_fraud_prob = model.predict_proba(input_data)[0][1] * 100
    final_risk_score = 99.99 if rule_triggered else ml_fraud_prob
    is_high_risk = final_risk_score >= (threshold * 100)

    st.session_state.evaluation_count += 1
    st.session_state.high_risk_count += int(is_high_risk)
    st.session_state.rule_alert_count += int(rule_triggered)
    logger.info(
        "transaction_evaluated model_version=%s type=%s risk_score=%.2f high_risk=%s rule_triggered=%s",
        model_version,
        tx_type,
        final_risk_score,
        is_high_risk,
        rule_triggered,
    )

    st.divider()
    st.subheader("Risk Score & Decision Output")

    res1, res2 = st.columns([1, 2])
    with res1:
        st.metric("Final Fraud Risk Score", f"{final_risk_score:.2f}%")
        if rule_triggered:
            st.caption("⚡ Intercepted by Dual-Side Ledger Rules Engine")

    with res2:
        if is_high_risk:
            st.error("🚨 HIGH RISK DETECTED — Transaction Blocked Immediately!")
            for reason in rule_reasons:
                st.warning(f"Ledger Audit Error: {reason}")
        else:
            st.success("🟢 LOW RISK — Transaction Approved Successfully")

st.divider()
st.subheader("Session monitoring")
monitor_columns = st.columns(3)
monitor_columns[0].metric("Evaluations", st.session_state.evaluation_count)
monitor_columns[1].metric("High-risk decisions", st.session_state.high_risk_count)
monitor_columns[2].metric("Ledger rule alerts", st.session_state.rule_alert_count)
st.caption(f"Model version: {model_version}. Counts are local to this browser session and reset when the session ends.")
