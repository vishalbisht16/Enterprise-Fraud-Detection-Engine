import unittest

from Streamlit_App.transaction_logic import (
    build_model_features,
    ledger_mismatch_reasons,
    validate_transaction_inputs,
)


class TransactionLogicTests(unittest.TestCase):
    def test_sender_only_transaction_has_no_receiver_features_or_mismatch(self):
        validate_transaction_inputs("DEBIT", 100, 200, 100)
        self.assertEqual(ledger_mismatch_reasons("DEBIT", 100, 200, 100, 40, 900), [])

        features = build_model_features("DEBIT", 100, 200, 100, 40, 900)
        self.assertEqual(features["oldbalanceDest"], 0.0)
        self.assertEqual(features["newbalanceDest"], 0.0)
        self.assertEqual(features["errorBalanceDest"], 0.0)

    def test_cash_in_uses_incoming_sender_balance_rule(self):
        validate_transaction_inputs("CASH_IN", 100, 200, 300)
        self.assertEqual(ledger_mismatch_reasons("CASH_IN", 100, 200, 300), [])
        self.assertNotEqual(ledger_mismatch_reasons("CASH_IN", 100, 200, 250), [])

    def test_payment_and_transfer_validate_receiver_balance(self):
        for tx_type in ("PAYMENT", "TRANSFER"):
            with self.subTest(tx_type=tx_type):
                validate_transaction_inputs(tx_type, 100, 200, 100, 50, 150)
                self.assertEqual(ledger_mismatch_reasons(tx_type, 100, 200, 100, 50, 150), [])
                self.assertTrue(ledger_mismatch_reasons(tx_type, 100, 200, 100, 50, 120))

    def test_invalid_amount_is_rejected(self):
        for amount in (0, -1, float("nan"), float("inf")):
            with self.subTest(amount=amount):
                with self.assertRaises(ValueError):
                    validate_transaction_inputs("DEBIT", amount, 200, 100)

    def test_receiver_balances_are_required_only_for_payment_and_transfer(self):
        with self.assertRaisesRegex(ValueError, "Receiver old balance"):
            validate_transaction_inputs("TRANSFER", 100, 200, 100)
        validate_transaction_inputs("CASH_OUT", 100, 200, 100)


if __name__ == "__main__":
    unittest.main()