#!/usr/bin/env python3
"""
Basic GraphRAG Functionality Test
Tests core components without requiring full GraphRAG installation
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Test our local implementations
def test_file_operations():
    """Test basic file operations"""
    print("[INFO] Testing file operations...")
    
    # Check input files exist
    input_dir = Path("./input")
    if not input_dir.exists():
        print("[ERROR] Input directory not found")
        return False
        
    input_files = list(input_dir.glob("*.txt"))
    if not input_files:
        print("[ERROR] No input files found")
        return False
        
    print(f"[OK] Found {len(input_files)} input files:")
    for file in input_files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    return True

def test_gemini_provider():
    """Test Gemini LLM provider"""
    print("[INFO] Testing Gemini provider...")
    
    try:
        from gemini_llm_provider import create_gemini_llm, GeminiLLMProvider
        
        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not set")
            return False
            
        # Create provider (don't actually call API)
        provider = create_gemini_llm({
            "api_key": api_key,
            "model": "gemini-1.5-pro-002",
        })
        
        print(f"[OK] Gemini provider created with model: {provider.config.model}")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error creating provider: {e}")
        return False

def test_pipeline_structure():
    """Test GraphRAG pipeline structure"""
    print("[INFO] Testing pipeline structure...")
    
    try:
        from graphrag_pipeline import GraphRAGConfig, GraphRAGPipeline
        
        # Create configuration
        config = GraphRAGConfig(
            input_dir="./input",
            output_dir="./output",
            cache_dir="./cache",
        )
        
        print(f"[OK] Configuration created:")
        print(f"  - Input: {config.input_dir}")
        print(f"  - Output: {config.output_dir}")
        print(f"  - Cache: {config.cache_dir}")
        print(f"  - Chunk size: {config.chunk_size}")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error creating config: {e}")
        return False

def test_search_engines():
    """Test search engine structure"""
    print("[INFO] Testing search engines...")
    
    try:
        from graphrag_search import LocalSearch, GlobalSearch, HybridSearch
        print("[OK] Search engine classes imported successfully")
        
        # Test factory function
        from graphrag_search import create_search_engines
        print("[OK] Factory function available")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error with search engines: {e}")
        return False

def test_server_structure():
    """Test server structure"""
    print("[INFO] Testing server structure...")
    
    try:
        # Import server components
        sys.path.append(".")
        
        # Test configuration loading
        import yaml
        config_path = "./graphrag_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] Configuration loaded with {len(config)} sections")
        else:
            print("[ERROR] Configuration file not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing server: {e}")
        return False

async def main():
    """Main test function"""
    print("GraphRAG Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("File Operations", test_file_operations),
        ("Gemini Provider", test_gemini_provider),
        ("Pipeline Structure", test_pipeline_structure),
        ("Search Engines", test_search_engines),
        ("Server Structure", test_server_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        try:
            if test_func():
                print(f"[PASS] {test_name}")
                passed += 1
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[FAIL] {test_name} - Exception: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] All basic functionality tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full pipeline test: python graphrag_pipeline.py")
        print("3. Start server: python graphrag_server.py")
        return 0
    else:
        print("[WARNING] Some tests failed. Check error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)