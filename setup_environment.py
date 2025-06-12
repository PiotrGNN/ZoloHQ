import os
from pathlib import Path


def setup_environment():
    """Setup environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"

    if env_file.exists():
        print(f"[INFO] Loading environment from {env_file}")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        os.environ[key] = value
                        print(
                            f"[ENV] {key}=***"
                            if "API" in key or "SECRET" in key or "TOKEN" in key
                            else f"[ENV] {key}={value}"
                        )
    else:
        print(f"[WARNING] .env file not found at {env_file}")
        print("[INFO] Using default environment variables")

        # Set default production environment
        defaults = {
            "TRADING_MODE": "production",
            "BYBIT_PRODUCTION_ENABLED": "true",
            "BYBIT_PRODUCTION_CONFIRMED": "true",
            "BYBIT_TESTNET": "false",
            "API_TIMEOUT": "30",
            "LOG_LEVEL": "INFO",
        }

        for key, value in defaults.items():
            if key not in os.environ:
                os.environ[key] = value
                print(f"[DEFAULT] {key}={value}")


if __name__ == "__main__":
    setup_environment()
    print("[SUCCESS] Environment configuration complete")
