#!/usr/bin/env python3
"""
Minimal Dependency Installation for GraphRAG
Installs only essential packages to get the system running
"""

import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Essential packages for basic functionality
ESSENTIAL_PACKAGES = [
    "google-generativeai==0.8.0",  # Gemini API
    "pydantic==2.4.2",             # Data validation
    "fastapi==0.104.1",            # Web framework
    "uvicorn[standard]==0.24.0",   # ASGI server
    "python-dotenv==1.0.0",        # Environment variables
    "pyyaml==6.0.1",               # YAML configuration
    "aiohttp==3.9.5",              # Async HTTP client
    "pandas==2.0.3",               # Data processing
    "numpy==1.24.4",               # Numerical computing
    "tiktoken==0.7.0",             # Token counting
]

# Optional packages for enhanced functionality
OPTIONAL_PACKAGES = [
    "sentence-transformers==2.2.2", # Text embeddings
    "faiss-cpu==1.7.4",            # Vector search
    "networkx==3.1",               # Graph processing
    "watchdog==3.0.0",             # File monitoring
    "websockets==12.0",            # WebSocket support
]

def install_package(package_name: str) -> bool:
    """Install a single package with error handling"""
    try:
        logger.info(f"Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully installed {package_name}")
            return True
        else:
            logger.error(f"❌ Failed to install {package_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout installing {package_name}")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing {package_name}: {e}")
        return False

def check_package_availability(package_name: str) -> bool:
    """Check if package is already installed"""
    try:
        # Extract package name without version
        pkg_name = package_name.split("==")[0].split(">=")[0].split("<=")[0]
        __import__(pkg_name.replace("-", "_"))
        return True
    except ImportError:
        return False

def install_packages(packages: list, required: bool = True) -> dict:
    """Install a list of packages"""
    results = {"success": [], "failed": [], "skipped": []}
    
    for package in packages:
        pkg_name = package.split("==")[0]
        
        # Check if already installed
        if check_package_availability(pkg_name):
            logger.info(f"⏭️  {pkg_name} already installed")
            results["skipped"].append(package)
            continue
        
        # Install package
        if install_package(package):
            results["success"].append(package)
        else:
            results["failed"].append(package)
            
            if required:
                logger.error(f"❌ Required package {package} failed to install")
            else:
                logger.warning(f"⚠️  Optional package {package} failed to install")
    
    return results

def main():
    """Main installation function"""
    logger.info("🚀 Starting minimal GraphRAG dependency installation...")
    
    # Install essential packages
    logger.info("📦 Installing essential packages...")
    essential_results = install_packages(ESSENTIAL_PACKAGES, required=True)
    
    # Install optional packages
    logger.info("📦 Installing optional packages...")
    optional_results = install_packages(OPTIONAL_PACKAGES, required=False)
    
    # Combine results
    total_success = len(essential_results["success"]) + len(optional_results["success"])
    total_failed = len(essential_results["failed"]) + len(optional_results["failed"])
    total_skipped = len(essential_results["skipped"]) + len(optional_results["skipped"])
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 INSTALLATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully installed: {total_success}")
    logger.info(f"⏭️  Already installed: {total_skipped}")
    logger.info(f"❌ Failed to install: {total_failed}")
    
    if essential_results["failed"]:
        logger.error("\n❌ CRITICAL: Essential packages failed to install:")
        for pkg in essential_results["failed"]:
            logger.error(f"  - {pkg}")
        logger.error("\nThe system may not work properly without these packages.")
        return 1
    
    if optional_results["failed"]:
        logger.warning("\n⚠️  Optional packages failed to install:")
        for pkg in optional_results["failed"]:
            logger.warning(f"  - {pkg}")
        logger.warning("\nSome features may be limited without these packages.")
    
    logger.info("\n🎉 Installation completed!")
    logger.info("\nNext steps:")
    logger.info("1. Run: python test_basic_functionality.py")
    logger.info("2. Run: python realtime_indexing.py --test-mode")
    logger.info("3. Run: python graphrag_server.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())