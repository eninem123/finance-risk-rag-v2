import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.finance_risk_rag.config import get_config

def verify():
    print("Verifying configuration...")
    config = get_config()
    print(f"Base Directory: {config.base_dir}")
    print(f"LLM Provider: {config.llm_provider}")
    print(f"Docs Directory: {config.docs_dir}")
    print(f"Is absolute: {config.docs_dir.is_absolute()}")

    # Check if env alias works (mocking environment)
    import os
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    from src.finance_risk_rag import config as config_mod
    config_mod._config = None # Reset singleton
    config2 = get_config()
    print(f"LLM API Key (from OPENAI_API_KEY): {config2.llm_api_key}")

    if config2.llm_api_key == "test-key-123":
        print("✅ Environment alias verification passed.")
    else:
        print("❌ Environment alias verification failed.")

if __name__ == "__main__":
    verify()
