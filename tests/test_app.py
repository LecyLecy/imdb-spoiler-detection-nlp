import unittest

from app import SAFE_EXAMPLE, SPOILER_EXAMPLE, load_components, predict_spoiler


class SpoilerPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline, _ = load_components()

    def test_safe_example_is_classified_as_non_spoiler(self) -> None:
        result = predict_spoiler(SAFE_EXAMPLE, self.pipeline)

        self.assertFalse(result["is_spoiler"])
        self.assertGreaterEqual(result["safe_probability"], 0.5)
        self.assertEqual(result["word_count"], len(SAFE_EXAMPLE.split()))

    def test_spoiler_example_is_classified_as_spoiler(self) -> None:
        result = predict_spoiler(SPOILER_EXAMPLE, self.pipeline)

        self.assertTrue(result["is_spoiler"])
        self.assertGreaterEqual(result["spoiler_probability"], 0.5)
        self.assertEqual(result["word_count"], len(SPOILER_EXAMPLE.split()))


if __name__ == "__main__":
    unittest.main()
