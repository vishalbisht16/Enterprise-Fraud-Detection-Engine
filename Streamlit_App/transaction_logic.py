import math


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