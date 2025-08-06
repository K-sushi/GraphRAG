#!/usr/bin/env python3
"""
Simple Bitcoin Price Test for GraphRAG System
テスト用：Bitcoinの現在価格検索

Gemini Web検索機能をテストし、Perplexity風システムの動作を確認します。
"""

import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# 環境変数から.envファイルを読み込み
from dotenv import load_dotenv
load_dotenv()

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """環境設定の確認"""
    print("🔍 環境設定の確認中...")
    
    # API キーの確認
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY が設定されていません")
        print("   .envファイルに以下を追加してください:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        return False
    
    if api_key == "YOUR_ACTUAL_API_KEY_HERE":
        print("❌ GEMINI_API_KEYが仮の値のままです")
        print("   .envファイルで実際のAPIキーに置き換えてください")
        return False
    
    print(f"✅ GEMINI_API_KEY設定済み (長さ: {len(api_key)})")
    
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
        print(f"❌ 必要なファイルが見つかりません: {missing_files}")
        return False
    
    print("✅ 必要なファイルが揃っています")
    return True

def test_imports():
    """必要なモジュールのインポートテスト"""
    print("📦 モジュールインポートテスト中...")
    
    try:
        import google.generativeai as genai
        print("✅ google-generativeai インポート成功")
    except ImportError as e:
        print(f"❌ google-generativeai インポートエラー: {e}")
        return False
    
    try:
        from gemini_web_search import GeminiWebSearchProvider
        print("✅ gemini_web_search インポート成功")
    except ImportError as e:
        print(f"❌ gemini_web_search インポートエラー: {e}")
        print("   ローカルのgemini_web_search.pyファイルを確認してください")
        return False
    
    return True

async def test_basic_gemini_connection():
    """Gemini APIの基本接続テスト"""
    print("🔗 Gemini API接続テスト中...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # 基本的なモデル作成テスト
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # 簡単なテキスト生成
        response = await asyncio.to_thread(
            model.generate_content,
            "Hello! Please respond with just 'API connection successful'"
        )
        
        print(f"✅ Gemini API接続成功")
        print(f"   レスポンス: {response.text[:100]}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API接続エラー: {e}")
        return False

async def test_web_search():
    """Web検索機能のテスト（Bitcoin価格）"""
    print("🔍 Gemini Web検索テスト中...")
    
    try:
        from gemini_web_search import GeminiWebSearchProvider
        
        # Web検索プロバイダーの作成
        api_key = os.getenv("GEMINI_API_KEY")
        config = {
            "search_model": "gemini-2.0-flash-exp",
            "max_search_results": 3,
            "cache_duration": 60
        }
        
        web_search = GeminiWebSearchProvider(api_key, config)
        
        # Bitcoin価格検索
        query = "現在のBitcoin価格は？"
        print(f"   クエリ: {query}")
        
        # 鮮度分析
        freshness_analysis = await web_search.analyze_query_freshness(query)
        print(f"   鮮度分析: {freshness_analysis['requires_web_search']} (信頼度: {freshness_analysis['confidence']:.2f})")
        
        # Web検索実行
        if freshness_analysis['requires_web_search']:
            search_result = await web_search.perform_web_search(query)
            print(f"✅ Web検索成功")
            print(f"   レスポンス長: {len(search_result['response'])}")
            print(f"   ソース数: {len(search_result['sources'])}")
            print(f"   最初の100文字: {search_result['response'][:100]}...")
            
            # ソースの表示
            if search_result['sources']:
                print("   主要ソース:")
                for i, source in enumerate(search_result['sources'][:3]):
                    print(f"     {i+1}. {source.get('title', 'Unknown')}")
                    if source.get('url'):
                        print(f"        URL: {source['url']}")
            
            return True
        else:
            print("⚠️ 検索は不要と判断されました")
            return True
            
    except Exception as e:
        print(f"❌ Web検索エラー: {e}")
        return False

async def run_bitcoin_test():
    """Bitcoin価格検索の包括的テスト"""
    print(f"\n{'='*60}")
    print("🪙 Bitcoin価格検索テスト - GraphRAG Perplexity風システム")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 段階的テスト実行
    test_steps = [
        ("環境設定確認", check_environment),
        ("モジュールインポート", test_imports),
        ("Gemini API接続", test_basic_gemini_connection),
        ("Web検索機能", test_web_search),
    ]
    
    results = {}
    
    for step_name, test_func in test_steps:
        print(f"\n🧪 ステップ {len([r for r in results.values() if r]) + 1}: {step_name}")
        print("-" * 40)
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        
        results[step_name] = result
        
        if not result:
            print(f"\n❌ テスト失敗: {step_name}")
            print("   このステップを修正してから続行してください。")
            break
        
        print(f"✅ {step_name} 完了")
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("📊 テスト結果サマリー")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"✅ 成功: {passed}/{total} ステップ")
    print(f"❌ 失敗: {total - passed}/{total} ステップ")
    
    if passed == total:
        print("\n🎉 全テスト成功！")
        print("   GraphRAG Perplexity風システムは正常に動作しています。")
        print("   次はtest_perplexity_system.pyで完全なデモを実行できます。")
    else:
        print("\n⚠️ いくつかのテストが失敗しました。")
        print("   失敗したステップを確認して修正してください。")
    
    return passed == total

async def main():
    """メイン実行関数"""
    try:
        success = await run_bitcoin_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ テスト中断")
        return 1
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        logger.exception("Unexpected error during testing")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)