from __future__ import annotations

import unittest

import numpy as np

from src.benchmarks.selection import (
    best_matches, content_tokens, min_statement_coverage, sample_questions, select_window, similarity_deciles,
)


class SelectionTests(unittest.TestCase):
    def test_best_matches_picks_most_similar_chunk(self) -> None:
        chunks = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        evidence = np.array([[0.1, 0.9], [1.0, 0.1]])
        index, sim = best_matches(chunks, evidence)
        self.assertEqual(list(index), [1, 0])
        self.assertTrue(all(0.9 < s <= 1.0 for s in sim))

    def _setup(self):
        # 10 chunks; types A/B concentrate in chunks 6-8, noise elsewhere (low similarity).
        index = [6, 7, 8, 6, 7, 1, 2, 9]
        sim = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.9]
        types = ["A", "A", "A", "B", "B", "A", "B", "B"]
        ids = [f"q{i}" for i in range(8)]
        return ids, types, index, sim

    def test_window_covers_densest_region_and_respects_min_similarity(self) -> None:
        ids, types, index, sim = self._setup()
        window = select_window(index, sim, types, chunk_count=10, window_chunks=3, per_type_target=3, min_similarity=0.5)
        self.assertEqual((window.start_chunk, window.end_chunk), (6, 9))
        self.assertEqual(window.per_type_available, {"A": 3, "B": 2})
        self.assertEqual(window.score, 5)  # min(3,3) + min(2,3)

    def test_tie_between_equal_score_windows_goes_left_example(self) -> None:
        index, sim, types = [6, 7, 6, 7], [0.9] * 4, ["A", "A", "B", "B"]
        window = select_window(index, sim, types, chunk_count=10, window_chunks=3, per_type_target=2, min_similarity=0.5)
        self.assertEqual(window.start_chunk, 5)  # [5,8) and [6,9) both score 4; leftmost wins

    def test_window_ties_resolve_to_leftmost(self) -> None:
        window = select_window([0, 5], [0.9, 0.9], ["A", "A"], chunk_count=10, window_chunks=2, per_type_target=1, min_similarity=0.5)
        self.assertEqual(window.start_chunk, 0)

    def test_invalid_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_window([0], [1.0], ["A"], chunk_count=5, window_chunks=6, per_type_target=1, min_similarity=0.5)

    def test_stratified_sample_is_seeded_inside_window_and_reports_shortfall(self) -> None:
        ids, types, index, sim = self._setup()
        window = select_window(index, sim, types, chunk_count=10, window_chunks=3, per_type_target=3, min_similarity=0.5)
        kwargs = dict(window=window, per_type_target=3, min_similarity=0.5, seed=7)
        chosen, shortfall = sample_questions(ids, types, index, sim, **kwargs)
        again, _ = sample_questions(ids, types, index, sim, **kwargs)
        self.assertEqual(chosen, again)  # deterministic
        self.assertEqual(shortfall, {"A": 0, "B": 1})  # only 2 B questions qualify
        in_window = {"q0", "q1", "q2", "q3", "q4"}
        self.assertTrue(set(chosen) <= in_window)
        self.assertNotIn("q5", chosen)  # outside window
        self.assertNotIn("q6", chosen)  # below min similarity

    def test_eligible_ids_restrict_the_sample_pool(self) -> None:
        ids, types, index, sim = self._setup()
        window = select_window(index, sim, types, chunk_count=10, window_chunks=3, per_type_target=3, min_similarity=0.5)
        chosen, shortfall = sample_questions(ids, types, index, sim, window=window, per_type_target=3,
                                             min_similarity=0.5, seed=1, eligible_ids={"q0", "q3"})
        self.assertEqual(sorted(chosen), ["q0", "q3"])
        self.assertEqual(shortfall, {"A": 2, "B": 2})

    def test_content_tokens_drop_stopwords_and_short_words(self) -> None:
        self.assertEqual(content_tokens("The risk of lung cancer is high"), ["risk", "lung", "cancer", "high"])

    def test_min_statement_coverage_uses_the_weakest_statement(self) -> None:
        words = {"mohs", "surgery", "preferred", "high-risk", "cscc"}
        full = "Mohs surgery is preferred for high-risk CSCC."
        self.assertEqual(min_statement_coverage(full, words), 1.0)
        mixed = full + "; Radiation always cures everything."
        self.assertEqual(min_statement_coverage(mixed, words), 0.0)
        self.assertEqual(min_statement_coverage("", words), 1.0)

    def test_deciles_have_nine_values(self) -> None:
        self.assertEqual(len(similarity_deciles(list(np.linspace(0, 1, 50)))), 9)


if __name__ == "__main__":
    unittest.main()
