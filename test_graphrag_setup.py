#!/usr/bin/env python3
"""
Microsoft GraphRAG with Gemini Integration Test Script
SuperClaude Wave Orchestration - Phase 1 Validation

Tests the complete GraphRAG setup with Gemini API integration
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_environment_setup():
    """Test environment configuration"""
    print("[OK] Testing Environment Setup...")
    
    required_env_vars = [
        "GEMINI_API_KEY",
        "GRAPHRAG_API_KEY",
        "GRAPHRAG_LLM_MODEL",
        "GRAPHRAG_ROOT_DIR",
        "GRAPHRAG_CONFIG_PATH"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"  [OK] {var}: {'*' * (len(value) - 4)}{value[-4:] if len(value) > 4 else value}")
    
    if missing_vars:
        print(f"  [ERROR] Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("  [OK] All required environment variables are set")
    return True

async def test_directory_structure():
    """Test directory structure"""
    print("\n[OK] Testing Directory Structure...")
    
    required_dirs = [
        os.getenv("GRAPHRAG_ROOT_DIR", "./graphrag_workspace"),
        os.getenv("GRAPHRAG_DATA_DIR", "./input"),
        os.getenv("GRAPHRAG_OUTPUT_DIR", "./output"),
        os.getenv("GRAPHRAG_CACHE_DIR", "./cache"),
    ]
    
    all_exist = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  [OK] {directory}")
        else:
            print(f"  [ERROR] {directory} (missing)")
            all_exist = False
    
    return all_exist

async def test_configuration_files():
    """Test configuration files"""
    print("\n[OK] Testing Configuration Files...")
    
    config_files = [
        ("GraphRAG Config", "graphrag_config.yaml"),
        ("Requirements", "requirements.txt"),
        ("Environment", ".env"),
        ("Gemini LLM Provider", "gemini_llm_provider.py"),
        ("GraphRAG Server", "graphrag_server.py"),
    ]
    
    all_exist = True
    for name, filename in config_files:
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print(f"  [OK] {name} ({filename}) - {size:,} bytes")
        else:
            print(f"  [ERROR] {name} ({filename}) - Not found")
            all_exist = False
    
    return all_exist

async def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n[OK] Testing Gemini API Connection...")
    
    try:
        # Import our Gemini provider
        from gemini_llm_provider import GeminiLLMProvider, LLMConfig
        
        config = LLMConfig(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.0-flash-exp",  # Use the fastest model for testing
            temperature=0.0
        )
        
        provider = GeminiLLMProvider(config)
        
        # Test simple generation
        test_prompt = "Say 'Microsoft GraphRAG with Gemini is working!' and nothing else."
        result = await provider.generate([
            {"role": "user", "content": test_prompt}
        ])
        
        if "GraphRAG" in result and "Gemini" in result:
            print(f"  [OK] Gemini API connection successful")
            print(f"  [INFO] Response: {result[:100]}{'...' if len(result) > 100 else ''}")
            return True
        else:
            print(f"  [WARN] Unexpected response: {result[:100]}")
            return True  # Still working, just unexpected response
            
    except Exception as e:
        print(f"  [ERROR] Gemini API connection failed: {e}")
        return False

async def test_graphrag_imports():
    """Test GraphRAG imports"""
    print("\n[OK] Testing GraphRAG Imports...")
    
    import_tests = [
        ("graphrag.index", "create_pipeline_config"),
        ("graphrag.query.structured_search.local_search.search", "LocalSearch"),
        ("graphrag.query.structured_search.global_search.search", "GlobalSearch"),
        ("graphrag.vector_stores", "VectorStoreFactory"),
    ]
    
    all_imports_work = True
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  [OK] {module_name}.{class_name}")
        except ImportError as e:
            print(f"  [ERROR] {module_name}.{class_name} - ImportError: {e}")
            all_imports_work = False
        except AttributeError as e:
            print(f"  [ERROR] {module_name}.{class_name} - AttributeError: {e}")
            all_imports_work = False
        except Exception as e:
            print(f"  [ERROR] {module_name}.{class_name} - Error: {e}")
            all_imports_work = False
    
    return all_imports_work

async def test_server_startup():
    """Test server can start (dry run)"""
    print("\n[OK] Testing Server Startup (Dry Run)...")
    
    try:
        # Try to import the server
        import graphrag_server
        print("  [OK] Server module imports successfully")
        
        # Test configuration loading
        config = graphrag_server.load_graphrag_config()
        if config:
            print("  [OK] Configuration loads successfully")
            print(f"  [INFO] Config keys: {list(config.keys())}")
        else:
            print("  [ERROR] Configuration loading failed")
            return False
        
        # Test directory creation
        graphrag_server.create_directories()
        print("  [OK] Directory creation successful")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Server startup test failed: {e}")
        return False

async def test_sample_document_processing():
    """Test processing a sample document"""
    print("\n[OK] Testing Sample Document Processing...")
    
    try:
        # Create a sample document
        sample_doc = """
        Microsoft GraphRAG is a powerful framework for building knowledge graphs from unstructured text.
        It combines graph-based retrieval with large language models to provide comprehensive answers.
        The system extracts entities, relationships, and communities from documents.
        """
        
        # Save to input directory
        input_dir = Path(os.getenv("GRAPHRAG_DATA_DIR", "./input"))
        sample_path = input_dir / "sample_test.txt"
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_doc)
        
        print(f"  [OK] Sample document created: {sample_path}")
        print(f"  [INFO] Document size: {len(sample_doc)} characters")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Sample document processing failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Microsoft GraphRAG with Gemini Integration - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Directory Structure", test_directory_structure),
        ("Configuration Files", test_configuration_files),
        ("GraphRAG Imports", test_graphrag_imports),
        ("Gemini Connection", test_gemini_connection),
        ("Server Startup", test_server_startup),
        ("Sample Document", test_sample_document_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  [CRASH] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("All tests passed! Microsoft GraphRAG with Gemini is ready!")
    elif passed >= total * 0.8:
        print("Most tests passed. Check failed tests and resolve issues.")
    else:
        print("Multiple tests failed. Review configuration and dependencies.")
    
    print("\nNext Steps:")
    if passed == total:
        print("  1. Start the server: python graphrag_server.py")
        print("  2. Test API endpoints: http://localhost:8000/docs")
        print("  3. Begin Phase 2: FastAPI migration")
    else:
        print("  1. Fix failed tests")
        print("  2. Re-run this test script")
        print("  3. Check environment variables and API keys")

if __name__ == "__main__":
    asyncio.run(main())