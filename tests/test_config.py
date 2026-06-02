import unittest
import os
from config import Config

class TestConfig(unittest.TestCase):
    def test_default_paths(self):
        config = Config()
        self.assertTrue(config.base_dir.is_absolute())
        self.assertEqual(config.llm_provider, "moonshot")

    def test_env_override(self):
        os.environ["LLM_PROVIDER"] = "openai"
        config = Config()
        self.assertEqual(config.llm_provider, "openai")
        # Cleanup
        del os.environ["LLM_PROVIDER"]

    def test_validate(self):
        config = Config()
        # Should fail if no API key is set
        self.assertFalse(config.validate())

        config.llm_api_key = "test-key"
        self.assertTrue(config.validate())

if __name__ == "__main__":
    unittest.main()
