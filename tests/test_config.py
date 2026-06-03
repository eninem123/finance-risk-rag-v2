import unittest

from finance_risk_rag.config import Config


class TestConfig(unittest.TestCase):
    def test_paths(self):
        config = Config()
        self.assertTrue(config.base_dir.is_absolute())

    def test_property(self):
        config = Config()
        self.assertIn("risk_entities.json", str(config.risk_entities_path))


if __name__ == "__main__":
    unittest.main()
