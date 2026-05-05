#!/usr/bin/env python3
"""
Comprehensive test suite - runs all component tests
Run with: python3 run_all_tests.py [--local|--cloud|--both]
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run command: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SecondBrainAI component tests')
    parser.add_argument(
        'mode',
        choices=['local', 'cloud', 'both'],
        default='both',
        nargs='?',
        help='Which components to test'
    )
    parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='Skip slow tests (vision analysis)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SECONDBRAINAI COMPONENT TEST SUITE")
    print("="*60)
    
    # Base tests (always run)
    tests = {
        "ingest_logic": ("python3 test_ingest.py", "Ingest Logic (Media Conversion)"),
        "audio": ("python3 test_audio.py", "Audio Engine (Whisper)"),
    }
    
    # Local mode tests
    local_tests = {
        "vision_local": ("python3 test_vision_local.py", "Vision Engine (Local Ollama)"),
        "app_local": ("streamlit run app_local.py --headless --logger.level=error", "App (Local)"),
    }
    
    # Cloud mode tests
    cloud_tests = {
        "vision_cloud": ("python3 test_vision_cloud.py", "Vision Engine (Cloud Gemini)"),
        "app_cloud": ("streamlit run app.py --headless --logger.level=error", "App (Cloud)"),
    }
    
    # Determine which tests to run
    all_tests = tests.copy()
    
    if args.mode in ['local', 'both']:
        if not args.skip_slow:
            all_tests.update(local_tests)
        else:
            # Run only quick connection tests
            all_tests["vision_local_quick"] = (
                "python3 -c \"from vision_engine_local import check_ollama_models; "
                "m = check_ollama_models(); "
                "print('✅ Ollama connection OK' if not m.get('error') else '❌ Failed')\"",
                "Vision Engine Quick Check (Local)"
            )
    
    if args.mode in ['cloud', 'both']:
        if not args.skip_slow:
            all_tests.update(cloud_tests)
        else:
            # Run only quick API key check
            all_tests["vision_cloud_quick"] = (
                "python3 -c \"from vision_engine import _load_api_key; "
                "_load_api_key(); "
                "print('✅ API key loaded')\"",
                "Vision Engine Quick Check (Cloud)"
            )
    
    # Run all tests
    results = {}
    for test_name, (cmd, description) in all_tests.items():
        results[test_name] = run_command(cmd, description)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! SecondBrainAI is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
