#!/usr/bin/env python
"""Comprehensive test runner for DiffML.

This script runs all tests, validates new features, and generates a report.
"""

import sys
import subprocess
import time
from pathlib import Path
import json
from datetime import datetime


class TestRunner:
    """Comprehensive test runner."""

    def __init__(self):
        """Initialize test runner."""
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_pytest(self):
        """Run pytest suite."""
        print("\n" + "="*60)
        print("Running pytest suite...")
        print("="*60)

        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=short", "--color=yes"],
                capture_output=True,
                text=True
            )

            self.results['pytest'] = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }

            if result.returncode == 0:
                print("✅ All tests passed!")
            else:
                print(f"❌ Tests failed with code {result.returncode}")
                print(result.stdout)

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Error running pytest: {e}")
            self.results['pytest'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def validate_new_features(self):
        """Validate all new features."""
        print("\n" + "="*60)
        print("Validating new features...")
        print("="*60)

        features_to_test = [
            ("American Options", "python src/diffml/datasets_american.py"),
            ("Multi-Barrier Options", "python src/diffml/datasets_multibarrier.py"),
            ("GPU Optimization", "python src/diffml/gpu_optimization.py"),
            ("Benchmarking Suite", "python src/diffml/benchmarking.py --category simulation"),
            ("Config Validation", "python src/diffml/config_validation.py template digital"),
        ]

        all_passed = True
        for feature_name, command in features_to_test:
            print(f"\nTesting {feature_name}...")
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    print(f"  ✅ {feature_name} working correctly")
                    self.results[feature_name] = {'success': True}
                else:
                    print(f"  ❌ {feature_name} failed")
                    self.results[feature_name] = {
                        'success': False,
                        'error': result.stderr
                    }
                    all_passed = False

            except subprocess.TimeoutExpired:
                print(f"  ⏱️ {feature_name} timed out")
                self.results[feature_name] = {
                    'success': False,
                    'error': 'Timeout'
                }
                all_passed = False
            except Exception as e:
                print(f"  ❌ Error testing {feature_name}: {e}")
                self.results[feature_name] = {
                    'success': False,
                    'error': str(e)
                }
                all_passed = False

        return all_passed

    def run_integration_tests(self):
        """Run integration tests specifically."""
        print("\n" + "="*60)
        print("Running integration tests...")
        print("="*60)

        try:
            result = subprocess.run(
                ["pytest", "tests/test_integration_workflows.py", "-v"],
                capture_output=True,
                text=True
            )

            self.results['integration'] = {
                'success': result.returncode == 0,
                'return_code': result.returncode
            }

            if result.returncode == 0:
                print("✅ Integration tests passed!")
            else:
                print(f"❌ Integration tests failed")

            return result.returncode == 0

        except Exception as e:
            print(f"❌ Error running integration tests: {e}")
            self.results['integration'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def check_code_quality(self):
        """Check code quality with linters."""
        print("\n" + "="*60)
        print("Checking code quality...")
        print("="*60)

        checks = [
            ("MyPy Type Checking", ["mypy", "src/diffml", "--ignore-missing-imports"]),
            ("Ruff Linting", ["ruff", "check", "src/diffml"]),
        ]

        all_passed = True
        for check_name, command in checks:
            print(f"\n{check_name}...")
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    print(f"  ✅ {check_name} passed")
                    self.results[check_name] = {'success': True}
                else:
                    print(f"  ⚠️ {check_name} has issues")
                    print(result.stdout[:500])  # Show first 500 chars
                    self.results[check_name] = {
                        'success': False,
                        'issues': result.stdout
                    }
                    # Don't fail on linting issues

            except Exception as e:
                print(f"  ⚠️ Could not run {check_name}: {e}")
                self.results[check_name] = {
                    'success': False,
                    'error': str(e)
                }

        return all_passed

    def generate_report(self):
        """Generate test report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': self.end_time - self.start_time if self.end_time else None,
            'results': self.results,
            'summary': {
                'total_checks': len(self.results),
                'passed': sum(1 for r in self.results.values() if r.get('success', False)),
                'failed': sum(1 for r in self.results.values() if not r.get('success', False))
            }
        }

        # Save JSON report
        report_path = Path('test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total checks: {report['summary']['total_checks']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"\nDetailed report saved to: {report_path}")

        return report['summary']['failed'] == 0

    def run_all(self):
        """Run all tests and validations."""
        self.start_time = time.time()

        print("\n🧪 DiffML Comprehensive Test Suite")
        print("="*60)

        # Run all test suites
        pytest_passed = self.run_pytest()
        features_valid = self.validate_new_features()
        integration_passed = self.run_integration_tests()
        # Code quality is optional
        self.check_code_quality()

        self.end_time = time.time()

        # Generate report
        all_passed = self.generate_report()

        if all_passed:
            print("\n✅ All critical tests passed! Ready for deployment.")
            return 0
        else:
            print("\n❌ Some tests failed. Please review the report.")
            return 1


if __name__ == "__main__":
    runner = TestRunner()
    sys.exit(runner.run_all())