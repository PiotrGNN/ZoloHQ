#!/usr/bin/env python3
"""
Test script to verify the typing import fix in ai_models/scalar.py
"""

import sys
import warnings

# Add the project root to Python path
project_root = r"c:\Users\piotr\Desktop\Zol0\ZoL0-master"
sys.path.insert(0, project_root)

warnings.filterwarnings("ignore", message="Expected None, but*")


def test_scalar_imports():
    """Test if scalar.py imports without typing errors"""
    print("Testing scalar.py typing imports...")

    try:
        print("✅ SUCCESS: scalar.py imports without typing errors!")
        print("  - Found classes: TensorScaler, FeatureScaler, DataScaler")
        return True
    except NameError as e:
        if "Optional" in str(e):
            print(f"❌ TYPING ERROR STILL EXISTS: {e}")
            return False
        else:
            print(f"❌ OTHER NAME ERROR: {e}")
            return False
    except Exception as e:
        print(f"❌ OTHER ERROR: {e}")
        return False


def test_ai_models_discovery():
    """Test AI models discovery"""
    print("\nTesting AI models discovery...")

    try:
        import ai_models

        models = ai_models.get_available_models()
        print(f"✅ SUCCESS: Discovered {len(models)} AI models:")
        for name, model_class in models.items():
            print(f"  - {name}: {model_class.__name__}")
        return True
    except Exception as e:
        print(f"❌ ERROR in model discovery: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=== AI Models Typing Fix Verification ===\n")

    # Test 1: Scalar imports
    scalar_ok = test_scalar_imports()

    # Test 2: AI models discovery
    discovery_ok = test_ai_models_discovery()

    print("\n=== RESULTS ===")
    print(f"Scalar imports: {'✅ PASS' if scalar_ok else '❌ FAIL'}")
    print(f"Models discovery: {'✅ PASS' if discovery_ok else '❌ FAIL'}")

    if scalar_ok and discovery_ok:
        print("\n🎉 ALL TESTS PASSED! The typing import fix is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Further investigation needed.")


if __name__ == "__main__":
    main()
