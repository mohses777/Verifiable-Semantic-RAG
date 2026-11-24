from __future__ import annotations

import unittest

from src.core.pricing import get_model_pricing


class PricingConfigTests(unittest.TestCase):
    def test_gpt_6_luna_rates_match_recorded_source(self) -> None:
        pricing = get_model_pricing("gpt-6-luna")
        self.assertEqual(pricing.input, 0.10)
        self.assertEqual(pricing.output, 0.50)
        # Q1 live QA usage: 5,140 prompt + 2,038 completion tokens.
        cost = pricing.input_cost(5140) + pricing.output_cost(2038)
        self.assertAlmostEqual(cost, 0.001533, places=6)

    def test_embedding_rate_and_unknown_model_is_rejected(self) -> None:
        self.assertEqual(get_model_pricing("text-embedding-3-large").input, 0.13)
        with self.assertRaises(KeyError):
            get_model_pricing("some-unconfigured-model")


if __name__ == "__main__":
    unittest.main()
