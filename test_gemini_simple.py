#!/usr/bin/env python3
"""
Simple Gemini API Test
Tests Gemini connectivity without heavy dependencies
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("✅ Environment variables loaded from .env file")
except ImportError:
    # Try to load manually if dotenv not available
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gemini_simple():
    """Test Gemini API with minimal setup"""
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment")
        return False
    
    logger.info(f"API Key found: ...{api_key[-4:]}")
    
    try:
        # Try direct HTTP request to Gemini API
        import json
        import aiohttp
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-002:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Say 'GraphRAG with Gemini is working!' and explain what GraphRAG is in one sentence."
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 100
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "candidates" in data and len(data["candidates"]) > 0:
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        logger.info(f"✅ Gemini Response: {content}")
                        return True
                    else:
                        logger.error(f"❌ Unexpected response format: {data}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"❌ HTTP {response.status}: {error_text}")
                    return False
                    
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Gemini: {e}")
        return False

async def test_system_components():
    """Test system components availability"""
    
    tests = []
    
    # Test 1: Environment variables
    required_vars = ["GEMINI_API_KEY", "GRAPHRAG_LLM_MODEL"]
    env_ok = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✅ {var}: ...{value[-10:] if len(value) > 10 else value}")
        else:
            logger.error(f"❌ {var}: Not set")
            env_ok = False
    
    tests.append(("Environment Variables", env_ok))
    
    # Test 2: File structure
    required_dirs = ["input", "output", "cache"]
    file_ok = True
    for dirname in required_dirs:
        path = f"./{dirname}"
        if os.path.exists(path):
            file_count = len(os.listdir(path)) if os.path.isdir(path) else 0
            logger.info(f"✅ {dirname}/: {file_count} files")
        else:
            logger.error(f"❌ {dirname}/: Not found")
            file_ok = False
    
    tests.append(("File Structure", file_ok))
    
    # Test 3: Configuration
    config_ok = True
    config_files = ["graphrag_config.yaml", ".env"]
    for config_file in config_files:
        if os.path.exists(config_file):
            size = os.path.getsize(config_file)
            logger.info(f"✅ {config_file}: {size} bytes")
        else:
            logger.error(f"❌ {config_file}: Not found")
            config_ok = False
    
    tests.append(("Configuration Files", config_ok))
    
    # Test 4: Python modules (basic)
    try:
        import json as json_mod
        import sys as sys_mod
        import asyncio as asyncio_mod
        logger.info("✅ Basic Python modules: Available")
        python_ok = True
    except ImportError as e:
        logger.error(f"❌ Basic Python modules: {e}")
        python_ok = False
    
    tests.append(("Python Modules", python_ok))
    
    return tests

async def main():
    """Main test function"""
    logger.info("🚀 Simple GraphRAG System Test")
    logger.info("=" * 50)
    
    # Test system components
    logger.info("📋 Testing System Components...")
    component_tests = await test_system_components()
    
    # Test Gemini API
    logger.info("\n🔗 Testing Gemini API...")
    gemini_test = await test_gemini_simple()
    
    # Summary
    total_tests = len(component_tests) + 1
    passed_tests = sum(1 for _, result in component_tests if result) + (1 if gemini_test else 0)
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in component_tests:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    gemini_status = "✅ PASS" if gemini_test else "❌ FAIL"
    logger.info(f"{gemini_status} Gemini API Test")
    
    success_rate = passed_tests / total_tests * 100
    logger.info(f"\n📈 Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 System is ready for basic operation!")
        logger.info("\nNext steps:")
        logger.info("1. Install remaining dependencies: pip install -r requirements.txt")
        logger.info("2. Test enhanced server: python enhanced_graphrag_server.py")
        logger.info("3. Run full test suite: python test_enhanced_system.py")
        return 0
    elif success_rate >= 60:
        logger.warning("⚠️  System partially working, some features may be limited")
        return 0
    else:
        logger.error("🚨 System has significant issues")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        sys.exit(1)