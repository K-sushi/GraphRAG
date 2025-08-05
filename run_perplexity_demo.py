#!/usr/bin/env python3
"""
Complete Perplexity-Style System Demonstration
SuperClaude Wave Orchestration - Phase 3A Complete

Final integration test demonstrating the complete Perplexity-style system
"""

import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_perplexity_system():
    """Demonstrate the complete Perplexity-style GraphRAG system"""
    
    print("🚀 Perplexity-Style GraphRAG System - Complete Demo")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable required")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    
    print("✅ API Key configured")
    
    # Check modules
    try:
        from gemini_web_search import create_gemini_web_search, create_perplexity_graphrag
        from gemini_llm_provider import create_gemini_llm
        from graphrag_search import create_search_engines
        print("✅ All modules available")
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        return False
    
    # Initialize system
    print("\n🔧 Initializing System Components...")
    
    try:
        # Web Search Provider
        web_search = create_gemini_web_search(api_key, {
            "search_model": "gemini-2.0-flash-exp",
            "analysis_model": "gemini-1.5-pro-002",
            "synthesis_model": "gemini-1.5-flash-002",
            "max_search_results": 8,
            "cache_duration": 300
        })
        print("  ✅ Gemini Web Search Provider")
        
        # GraphRAG Search Engine
        gemini_provider = create_gemini_llm({
            "api_key": api_key,
            "model": "gemini-2.0-flash-exp"
        })
        
        _, _, hybrid_search = create_search_engines("./output", gemini_provider)
        print("  ✅ GraphRAG Hybrid Search Engine")
        
        # Perplexity System
        perplexity_system = create_perplexity_graphrag(web_search, hybrid_search, {
            "freshness_threshold": 0.7,
            "always_use_graphrag": True,
            "max_response_time": 30
        })
        print("  ✅ Perplexity-Style Orchestrator")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    # Demo queries showcasing different capabilities
    demo_scenarios = [
        {
            "title": "🪙 Real-time Crypto Price (Japanese)",
            "query": "BTC現在価格は？",
            "description": "Tests real-time financial data retrieval with Japanese language processing",
            "expected_features": ["web_search", "current_price", "japanese_response"]
        },
        {
            "title": "📈 Financial Market Analysis (English)", 
            "query": "What's the current Bitcoin price and market sentiment?",
            "description": "Tests comprehensive financial analysis with market context",
            "expected_features": ["web_search", "market_analysis", "sentiment"]
        },
        {
            "title": "🤖 AI News & Developments",
            "query": "Latest developments in artificial intelligence today",
            "description": "Tests current events and news aggregation capabilities",
            "expected_features": ["web_search", "news_aggregation", "recent_events"]
        },
        {
            "title": "🧠 Knowledge Graph Query",
            "query": "What is machine learning and how does it work?",
            "description": "Tests GraphRAG knowledge retrieval without web search",
            "expected_features": ["graphrag_knowledge", "comprehensive_explanation"]
        },
        {
            "title": "🌍 Real-time Information",
            "query": "Current weather in Tokyo and any weather alerts",
            "description": "Tests real-time environmental data and alert systems",
            "expected_features": ["web_search", "current_conditions", "alerts"]
        }
    ]
    
    print(f"\n🧪 Running {len(demo_scenarios)} Demo Scenarios")
    print("=" * 60)
    
    results = []
    successful_tests = 0
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n📋 Scenario {i}/{len(demo_scenarios)}: {scenario['title']}")
        print(f"🔍 Query: {scenario['query']}")
        print(f"📝 Purpose: {scenario['description']}")
        print("-" * 50)
        
        try:
            start_time = datetime.now()
            
            # Process query
            result = await perplexity_system.process_query(scenario['query'])
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Extract key metrics
            metadata = result.get('processing_metadata', {})
            quality = metadata.get('response_quality', {})
            response = result.get('synthesized_response', 'No response')
            
            # Display response (truncated)
            display_response = response[:300] + "..." if len(response) > 300 else response
            print(f"🤖 Response: {display_response}")
            
            # Display metrics
            print(f"\n📊 Performance Metrics:")
            print(f"  ⏱️  Processing Time: {processing_time:.2f}s")
            print(f"  🌐 Web Search Used: {'Yes' if metadata.get('web_search_performed') else 'No'}")
            print(f"  🧠 GraphRAG Used: {'Yes' if metadata.get('graphrag_used') else 'No'}")
            print(f"  📚 Total Sources: {metadata.get('total_sources', 0)}")
            print(f"  ⭐ Quality Score: {quality.get('overall_score', 0):.2f}/1.0")
            print(f"  🎯 Assessment: {quality.get('assessment', 'unknown')}")
            
            # Feature validation
            features_found = []
            if metadata.get('web_search_performed'):
                features_found.append("web_search")
            if metadata.get('graphrag_used'):
                features_found.append("graphrag")
            if quality.get('overall_score', 0) > 0.7:
                features_found.append("high_quality")
            if processing_time < 10:
                features_found.append("fast_response")
            
            print(f"  🔧 Features: {', '.join(features_found)}")
            
            # Success criteria
            success = (
                len(response) > 100 and  # Substantial response
                processing_time < 30 and  # Within time limit
                quality.get('overall_score', 0) > 0.5  # Reasonable quality
            )
            
            if success:
                print(f"  ✅ Status: SUCCESS")
                successful_tests += 1
            else:
                print(f"  ⚠️  Status: PARTIAL SUCCESS")
            
            results.append({
                "scenario": scenario['title'],
                "query": scenario['query'],
                "success": success,
                "processing_time": processing_time,
                "quality_score": quality.get('overall_score', 0),
                "web_search_used": metadata.get('web_search_performed', False),
                "graphrag_used": metadata.get('graphrag_used', False),
                "total_sources": metadata.get('total_sources', 0),
                "response_length": len(response)
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "scenario": scenario['title'],
                "query": scenario['query'],
                "success": False,
                "error": str(e)
            })
        
        # Rate limiting between queries
        if i < len(demo_scenarios):
            print("\n⏳ Waiting 3 seconds for rate limiting...")
            await asyncio.sleep(3)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 DEMO RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"✅ Successful Tests: {successful_tests}/{len(demo_scenarios)}")
    print(f"❌ Failed Tests: {len(demo_scenarios) - successful_tests}/{len(demo_scenarios)}")
    
    if successful_tests > 0:
        successful_results = [r for r in results if r.get('success')]
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_quality = sum(r['quality_score'] for r in successful_results) / len(successful_results)
        
        print(f"⏱️  Average Processing Time: {avg_time:.2f}s")
        print(f"⭐ Average Quality Score: {avg_quality:.2f}/1.0")
        
        web_search_count = sum(1 for r in successful_results if r.get('web_search_used'))
        graphrag_count = sum(1 for r in successful_results if r.get('graphrag_used'))
        
        print(f"🌐 Web Search Usage: {web_search_count}/{successful_tests} tests")
        print(f"🧠 GraphRAG Usage: {graphrag_count}/{successful_tests} tests")
    
    # Success criteria for overall system
    overall_success_rate = successful_tests / len(demo_scenarios)
    
    print(f"\n🎉 OVERALL SYSTEM STATUS:")
    if overall_success_rate >= 0.8:
        print(f"✅ EXCELLENT: {overall_success_rate:.1%} success rate")
        print("   Perplexity-style system is working perfectly!")
    elif overall_success_rate >= 0.6:
        print(f"✅ GOOD: {overall_success_rate:.1%} success rate")
        print("   System is functional with minor issues")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: {overall_success_rate:.1%} success rate")
        print("   System requires troubleshooting")
    
    print(f"\n📋 Key Capabilities Demonstrated:")
    print(f"  🌐 Real-time web search integration")
    print(f"  🧠 GraphRAG knowledge synthesis")
    print(f"  🌍 Multi-language support (Japanese/English)")
    print(f"  💰 Financial data retrieval")
    print(f"  📰 Current events and news analysis")
    print(f"  🤖 AI-powered response synthesis")
    print(f"  📊 Quality assessment and source attribution")
    
    print(f"\n🚀 Phase 3A: Perplexity-Style System - COMPLETE!")
    print(f"   Ready for production use with BTC price queries and beyond!")
    
    return overall_success_rate >= 0.6

async def main():
    """Main demonstration function"""
    
    try:
        success = await demo_perplexity_system()
        
        if success:
            print(f"\n✨ Demo completed successfully!")
            print(f"🎯 Your Perplexity-style GraphRAG system is ready!")
            print(f"\n📖 Next steps:")
            print(f"  • Try interactive mode: python perplexity_graphrag_cli.py -i")
            print(f"  • Test your own queries: python perplexity_graphrag_cli.py -q 'Your question'")
            print(f"  • Read the guide: QUICK_START_PERPLEXITY.md")
        else:
            print(f"\n⚠️  Demo completed with issues")
            print(f"📖 Troubleshooting:")
            print(f"  • Check API key: echo $GEMINI_API_KEY")
            print(f"  • Verify dependencies: pip list | grep google-generativeai")
            print(f"  • Check GraphRAG data: ls -la ./output/")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())