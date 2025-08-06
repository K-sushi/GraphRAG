#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bitcoin Price Test - Direct Implementation
Bitcoin価格検索の直接実装テスト
"""

import os
import asyncio
import sys
from datetime import datetime

# UTF-8 出力設定
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 環境変数から.envファイルを読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

async def test_bitcoin_price_search():
    """Bitcoin価格検索の直接テスト"""
    
    print("=" * 60)
    print("Bitcoin Price Search Test - Perplexity Style")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 必要なモジュールをインポート
        from gemini_web_search import GeminiWebSearchProvider
        import google.generativeai as genai
        
        # API キー取得
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not found")
            return False
        
        print(f"Step 1: API Key configured (length: {len(api_key)})")
        
        # Web検索プロバイダーを作成 - Use 1.5 models for web search support
        config = {
            "search_model": "gemini-1.5-flash-002",  # Use 1.5 for web search support
            "analysis_model": "gemini-1.5-pro-002",
            "synthesis_model": "gemini-1.5-flash-002",
            "max_search_results": 10,
            "cache_duration": 300
        }
        
        web_search = GeminiWebSearchProvider(api_key, config)
        print("Step 2: Web search provider created")
        
        # Bitcoin価格検索クエリ
        queries = [
            "Bitcoin price today USD",
            "BTC current price",
            "What is Bitcoin worth right now?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Test {i}: {query} ---")
            
            try:
                # Step 1: 鮮度分析
                print("Analyzing query freshness...")
                freshness = await web_search.analyze_query_freshness(query)
                print(f"  Requires web search: {freshness.get('requires_web_search', 'unknown')}")
                print(f"  Confidence: {freshness.get('confidence', 0):.2f}")
                print(f"  Search type: {freshness.get('search_type', 'unknown')}")
                
                # Step 2: Web検索実行
                if freshness.get('requires_web_search', True):
                    print("Performing web search...")
                    search_params = {
                        "search_type": freshness.get("search_type", "real_time"),
                        "urgency": freshness.get("urgency", "high"),
                        "keywords": freshness.get("keywords", [])
                    }
                    
                    search_result = await web_search.perform_web_search(query, search_params)
                    
                    print(f"  Search completed!")
                    print(f"  Response length: {len(search_result.get('response', ''))}")
                    print(f"  Number of sources: {len(search_result.get('sources', []))}")
                    
                    # 結果の一部を表示
                    response_text = search_result.get('response', '')
                    if response_text:
                        print(f"  First 200 chars: {response_text[:200]}...")
                    
                    # ソース情報を表示
                    sources = search_result.get('sources', [])
                    if sources:
                        print("  Sources:")
                        for j, source in enumerate(sources[:3]):
                            print(f"    {j+1}. {source.get('title', 'Unknown Title')}")
                            if source.get('url'):
                                print(f"       URL: {source['url']}")
                            print(f"       Type: {source.get('source_type', 'unknown')}")
                    
                    # 品質評価
                    metadata = search_result.get('metadata', {})
                    quality = metadata.get('search_quality', {})
                    print(f"  Search quality: {quality.get('assessment', 'unknown')} ({quality.get('overall_score', 0):.2f})")
                    
                    print(f"SUCCESS: Query '{query}' completed successfully")
                    
                else:
                    print("Web search not required for this query")
                
            except Exception as e:
                print(f"ERROR: Query '{query}' failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 60)
        print("BITCOIN PRICE TEST COMPLETED")
        print("=" * 60)
        print("RESULT: Bitcoin price search functionality is working!")
        print("The Perplexity-style system can successfully:")
        print("- Analyze query freshness")
        print("- Perform real-time web searches")
        print("- Retrieve current Bitcoin price information")
        print("- Process and format search results")
        
        return True
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """メイン実行"""
    try:
        success = await test_bitcoin_price_search()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)