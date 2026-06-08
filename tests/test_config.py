import unittest

from finance_risk_rag.config import get_config


class TestConfig(unittest.TestCase):
    def test_singleton_config(self):
        c1 = get_config()
        c2 = get_config()
        self.assertIs(c1, c2)

    def test_default_paths(self):
        config = get_config()
        self.assertTrue(config.base_dir.is_absolute())
        self.assertEqual(config.docs_dir.name, "docs")


if __name__ == "__main__":
    unittest.main()
