#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Bitcoin Price Test for GraphRAG System
Bitcoin価格検索テスト用スクリプト
"""

import os
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 出力設定
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 環境変数から.envファイルを読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("OK: dotenv loaded")
except ImportError:
    print("WARNING: python-dotenv not available, using system environment")

def check_environment():
    """環境設定の確認"""
    print("=== Environment Check ===")
    
    # API キーの確認
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        print("Please add to .env file:")
        print("GEMINI_API_KEY=your_actual_api_key_here")
        return False
    
    if api_key == "YOUR_ACTUAL_API_KEY_HERE":
        print("ERROR: GEMINI_API_KEY is placeholder value")
        print("Please replace with actual API key in .env file")
        return False
    
    print(f"OK: GEMINI_API_KEY configured (length: {len(api_key)})")
    
    # 必要なファイルの確認
    required_files = [
        "gemini_web_search.py",
        "gemini_llm_provider.py", 
        "graphrag_search.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        return False
    
    print("OK: Required files found")
    return True

def test_imports():
    """必要なモジュールのインポートテスト"""
    print("=== Import Test ===")
    
    try:
        import google.generativeai as genai
        print("OK: google-generativeai imported")
    except ImportError as e:
        print(f"ERROR: google-generativeai import failed: {e}")
        return False
    
    try:
        from gemini_web_search import GeminiWebSearchProvider
        print("OK: gemini_web_search imported")
    except ImportError as e:
        print(f"ERROR: gemini_web_search import failed: {e}")
        return False
    
    return True

async def test_basic_gemini():
    """Gemini API基本テスト"""
    print("=== Gemini API Test ===")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        response = await asyncio.to_thread(
            model.generate_content,
            "Hello! Respond with: API connection successful"
        )
        
        print(f"OK: Gemini API connected")
        print(f"Response: {response.text[:50]}...")
        return True
        
    except Exception as e:
        print(f"ERROR: Gemini API test failed: {e}")
        return False

async def test_web_search_simple():
    """簡単なWeb検索テスト"""
    print("=== Web Search Test ===")
    
    try:
        from gemini_web_search import GeminiWebSearchProvider
        
        api_key = os.getenv("GEMINI_API_KEY")
        web_search = GeminiWebSearchProvider(api_key)
        
        query = "Bitcoin price today"
        print(f"Query: {query}")
        
        # 鮮度分析テスト
        freshness = await web_search.analyze_query_freshness(query)
        print(f"Freshness analysis: needs_web_search={freshness['requires_web_search']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Web search test failed: {e}")
        return False

async def main():
    """メイン実行"""
    print("=" * 50)
    print("Bitcoin Price Test - GraphRAG System")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    tests = [
        ("Environment Check", check_environment),
        ("Import Test", test_imports), 
        ("Gemini API Test", test_basic_gemini),
        ("Web Search Test", test_web_search_simple),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append(result)
            status = "PASS" if result else "FAIL"
            print(f"Result: {status}")
            
            if not result:
                print("Stopping tests due to failure")
                break
                
        except Exception as e:
            print(f"ERROR: Test failed with exception: {e}")
            results.append(False)
            break
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("SUCCESS: All tests passed!")
        print("System is ready for full Bitcoin price demo")
    else:
        print("FAILURE: Some tests failed")
        print("Please fix issues before proceeding")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        exit(1)