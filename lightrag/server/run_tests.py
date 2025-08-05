#!/usr/bin/env python3
"""
Test runner for LightRAG server with comprehensive test execution and reporting.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

def run_command(cmd, description=None):
    """Run a command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print(f"ERROR: {result.stderr}", file=sys.stderr)
            
        return result.returncode == 0
    
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}", file=sys.stderr)
        return False

def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "testing",
        "REQUIRE_AUTH": "false",
        "OPENAI_API_KEY": "test-key",
        "LIGHTRAG_API_KEYS": "test-api-key-1,test-api-key-2",
        "WORKING_DIR": "./test_cache",
        "HOST": "127.0.0.1",
        "PORT": "8001",  # Different from production
        "LOG_LEVEL": "DEBUG",
        "RATE_LIMIT_PER_MINUTE": "1000",  # High limits for testing
        "MAX_TOKENS_PER_MINUTE": "100000",
        "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
        "CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "5",
        "CORS_ORIGINS": "http://localhost:3000,http://localhost:8080",
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("Test environment configured")

def cleanup_test_artifacts():
    """Clean up test artifacts and cache."""
    artifacts_to_remove = [
        "test_cache",
        "htmlcov",
        "coverage.xml",
        ".coverage",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        ".mypy_cache"
    ]
    
    for artifact in artifacts_to_remove:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
                print(f"Removed directory: {artifact}")
            else:
                os.remove(artifact)
                print(f"Removed file: {artifact}")

def run_unit_tests(args):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest"]
    
    # Add test paths
    cmd.extend([
        "tests/test_config.py",
        "tests/test_utils.py", 
        "tests/test_models.py",
    ])
    
    # Add common options
    cmd.extend([
        "-v",
        "--tb=short",
        "-m", "not integration and not slow",
    ])
    
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    return run_command(cmd, "Unit Tests")

def run_integration_tests(args):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest"]
    
    # Add test paths
    cmd.extend([
        "tests/test_app.py",
        "tests/test_security.py",
    ])
    
    # Add common options
    cmd.extend([
        "-v",
        "--tb=short",
    ])
    
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term", "--cov-append"])
    
    return run_command(cmd, "Integration Tests")

def run_security_tests(args):
    """Run security-focused tests."""
    cmd = ["python", "-m", "pytest"]
    
    cmd.extend([
        "tests/test_security.py",
        "-v",
        "--tb=short",
        "-m", "security or auth",
    ])
    
    return run_command(cmd, "Security Tests")

def run_performance_tests(args):
    """Run performance tests."""
    if not args.performance:
        print("Skipping performance tests (use --performance to enable)")
        return True
    
    cmd = ["python", "-m", "pytest"]
    
    cmd.extend([
        "tests/",
        "-v",
        "--tb=short",
        "-m", "performance",
        "--benchmark-only",
        "--benchmark-warmup=on",
        "--benchmark-autosave",
    ])
    
    return run_command(cmd, "Performance Tests")

def run_code_quality_checks(args):
    """Run code quality checks."""
    checks = []
    
    # Type checking with mypy
    if shutil.which("mypy"):
        mypy_cmd = ["mypy", ".", "--ignore-missing-imports", "--strict-optional"]
        checks.append((mypy_cmd, "Type checking (mypy)"))
    
    # Linting with flake8
    if shutil.which("flake8"):
        flake8_cmd = ["flake8", ".", "--max-line-length=100", "--ignore=E203,W503"]
        checks.append((flake8_cmd, "Code linting (flake8)"))
    
    # Security scanning with bandit
    if shutil.which("bandit"):
        bandit_cmd = ["bandit", "-r", ".", "-x", "tests/,venv/"]
        checks.append((bandit_cmd, "Security scanning (bandit)"))
    
    # Dependency vulnerability check
    if shutil.which("safety"):
        safety_cmd = ["safety", "check"]
        checks.append((safety_cmd, "Dependency vulnerability check (safety)"))
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False
            if not args.continue_on_error:
                break
    
    return all_passed

def generate_test_report(args):
    """Generate comprehensive test report."""
    if not args.report:
        return True
    
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    # Coverage report
    if os.path.exists("htmlcov/index.html"):
        print("Coverage report generated: htmlcov/index.html")
    
    # Test results summary
    if os.path.exists(".pytest_cache"):
        print("Test cache available: .pytest_cache/")
    
    # Generate markdown report
    report_content = f"""
# LightRAG Server Test Report

## Test Configuration
- Environment: {os.environ.get('ENVIRONMENT', 'testing')}
- Python Version: {sys.version}
- Test Directory: {Path.cwd()}

## Test Results
- Unit Tests: {'✅ PASSED' if args.unit_passed else '❌ FAILED'}
- Integration Tests: {'✅ PASSED' if args.integration_passed else '❌ FAILED'}
- Security Tests: {'✅ PASSED' if args.security_passed else '❌ FAILED'}
- Code Quality: {'✅ PASSED' if args.quality_passed else '❌ FAILED'}

## Coverage
Coverage report available at: htmlcov/index.html

## Next Steps
1. Review any failing tests
2. Check coverage report for uncovered code
3. Address any security or quality issues
4. Run performance tests if needed
"""
    
    with open("test_report.md", "w") as f:
        f.write(report_content)
    
    print("Test report generated: test_report.md")
    return True

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="LightRAG Server Test Runner")
    
    # Test categories
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--quality", action="store_true", help="Run code quality checks")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test artifacts before running")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running tests after failures")
    
    # Pytest passthrough
    parser.add_argument("--pytest-args", help="Additional arguments to pass to pytest")
    
    args = parser.parse_args()
    
    # Set up environment
    setup_test_environment()
    
    # Clean up if requested
    if args.cleanup:
        cleanup_test_artifacts()
    
    # Determine what to run
    run_all = args.all or not any([args.unit, args.integration, args.security, args.performance, args.quality])
    
    # Track results for reporting
    args.unit_passed = True
    args.integration_passed = True
    args.security_passed = True
    args.performance_passed = True
    args.quality_passed = True
    
    all_passed = True
    
    try:
        # Run test categories
        if run_all or args.unit:
            print("\n🧪 Running Unit Tests...")
            args.unit_passed = run_unit_tests(args)
            all_passed &= args.unit_passed
        
        if run_all or args.integration:
            print("\n🔗 Running Integration Tests...")
            args.integration_passed = run_integration_tests(args)
            all_passed &= args.integration_passed
        
        if run_all or args.security:
            print("\n🔒 Running Security Tests...")
            args.security_passed = run_security_tests(args)
            all_passed &= args.security_passed
        
        if run_all or args.performance:
            print("\n🚀 Running Performance Tests...")
            args.performance_passed = run_performance_tests(args)
            all_passed &= args.performance_passed
        
        if run_all or args.quality:
            print("\n📊 Running Code Quality Checks...")
            args.quality_passed = run_code_quality_checks(args)
            all_passed &= args.quality_passed
        
        # Generate report
        generate_test_report(args)
        
        # Final results
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()