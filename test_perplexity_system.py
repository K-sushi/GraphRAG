#!/usr/bin/env python3
"""
Perplexity-Style System Demonstration
SuperClaude Wave Orchestration - Phase 3A

Test the complete Perplexity-style real-time search + AI reasoning system
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our components
try:
    from gemini_web_search import create_gemini_web_search, create_perplexity_graphrag
    from gemini_llm_provider import create_gemini_llm
    from graphrag_search import create_search_engines
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure all required modules are available")
    exit(1)

class PerplexitySystemDemo:
    """
    Demonstration system for Perplexity-style GraphRAG integration
    
    Tests real-world queries like BTC prices, current events, etc.
    """
    
    def __init__(self, data_dir: str = "./output"):
        self.data_dir = data_dir
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # System components
        self.web_search = None
        self.graphrag_search = None
        self.perplexity_system = None
        
        logger.info("Perplexity System Demo initialized")
    
    async def setup_system(self):
        """Initialize all system components"""
        logger.info("Setting up Perplexity-style system components...")
        
        try:
            # Create web search provider
            web_search_config = {
                "search_model": "gemini-2.0-flash-exp",  # Fast for search
                "analysis_model": "gemini-1.5-pro-002",  # Deep for analysis
                "synthesis_model": "gemini-1.5-flash-002",  # Balanced for synthesis
                "max_search_results": 10,
                "cache_duration": 300  # 5 minutes
            }
            
            self.web_search = create_gemini_web_search(self.api_key, web_search_config)
            logger.info("✅ Web search provider created")
            
            # Create GraphRAG search engines
            gemini_provider = create_gemini_llm({
                "api_key": self.api_key,
                "model": "gemini-2.0-flash-exp"
            })
            
            local_search, global_search, hybrid_search = create_search_engines(
                self.data_dir, gemini_provider
            )
            self.graphrag_search = hybrid_search
            logger.info("✅ GraphRAG search engines created")
            
            # Create Perplexity-style system
            perplexity_config = {
                "freshness_threshold": 0.7,
                "always_use_graphrag": True,
                "max_response_time": 30
            }
            
            self.perplexity_system = create_perplexity_graphrag(
                self.web_search, self.graphrag_search, perplexity_config
            )
            logger.info("✅ Perplexity-style system created")
            
            return True
            
        except Exception as e:
            logger.error(f"System setup failed: {e}")
            return False
    
    async def test_query(self, query: str, force_web_search: bool = False) -> dict:
        """Test a single query with the Perplexity system"""
        
        logger.info(f"🔍 Testing query: {query}")
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")
        
        try:
            start_time = datetime.now()
            
            # Process query
            result = await self.perplexity_system.process_query(
                query, force_web_search=force_web_search
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Display results
            print(f"\n📝 Response:")
            print(f"{result.get('synthesized_response', 'No response available')}")
            
            # Display metadata
            metadata = result.get('processing_metadata', {})
            print(f"\n📊 Processing Metadata:")
            print(f"  • Response Type: {metadata.get('response_type', 'unknown')}")
            print(f"  • Processing Time: {processing_time:.2f}s")
            print(f"  • Web Search Used: {metadata.get('web_search_performed', False)}")
            print(f"  • GraphRAG Used: {metadata.get('graphrag_used', False)}")
            print(f"  • Total Sources: {metadata.get('total_sources', 0)}")
            
            # Quality assessment
            quality = metadata.get('response_quality', {})
            print(f"  • Quality Score: {quality.get('overall_score', 0):.2f}/1.0")
            print(f"  • Quality Assessment: {quality.get('assessment', 'unknown')}")
            print(f"  • Web Sources: {quality.get('web_sources', 0)}")
            print(f"  • GraphRAG Sources: {quality.get('graphrag_sources', 0)}")
            
            # Display sources
            sources = result.get('all_sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources[:5]):  # Show top 5
                    print(f"  {i+1}. {source.get('title', 'Unknown Title')}")
                    if source.get('url'):
                        print(f"     🔗 {source['url']}")
                    print(f"     📑 Type: {source.get('source_type', 'unknown')}")
                    print(f"     🏷️  Origin: {source.get('source_origin', 'unknown')}")
                    if source.get('confidence'):
                        print(f"     📊 Confidence: {source['confidence']:.2f}")
                    print()
            
            # Freshness analysis
            freshness = metadata.get('freshness_analysis', {})
            if freshness:
                print(f"📅 Freshness Analysis:")
                print(f"  • Requires Web Search: {freshness.get('requires_web_search', False)}")
                print(f"  • Confidence: {freshness.get('confidence', 0):.2f}")
                print(f"  • Search Type: {freshness.get('search_type', 'unknown')}")
                print(f"  • Urgency: {freshness.get('urgency', 'unknown')}")
                print(f"  • Expected Freshness: {freshness.get('expected_freshness', 'unknown')}")
            
            return {
                "success": True,
                "query": query,
                "processing_time": processing_time,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Query test failed: {e}")
            print(f"\n❌ Error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    async def run_demo_suite(self):
        """Run a comprehensive demo of the Perplexity system"""
        
        # Demo queries covering different scenarios
        demo_queries = [
            {
                "query": "BTC現在価格は？",
                "description": "Real-time cryptocurrency price (Japanese)",
                "force_web": False
            },
            {
                "query": "What is the current Bitcoin price in USD?",
                "description": "Real-time cryptocurrency price (English)",
                "force_web": False
            },
            {
                "query": "Latest news about artificial intelligence today",
                "description": "Current events requiring web search",
                "force_web": False
            },
            {
                "query": "What is machine learning?",
                "description": "General knowledge query (GraphRAG should handle)",
                "force_web": False
            },
            {
                "query": "Current weather in Tokyo",
                "description": "Real-time information",
                "force_web": False
            },
            {
                "query": "What happened in the stock market today?",
                "description": "Time-sensitive financial information",
                "force_web": False
            }
        ]
        
        print(f"\n🚀 Starting Perplexity-Style System Demo")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💡 Testing {len(demo_queries)} different query types")
        
        results = []
        successful_tests = 0
        
        for i, test_case in enumerate(demo_queries, 1):
            print(f"\n🧪 Test {i}/{len(demo_queries)}: {test_case['description']}")
            
            result = await self.test_query(
                test_case["query"], 
                force_web_search=test_case["force_web"]
            )
            
            results.append(result)
            if result["success"]:
                successful_tests += 1
            
            # Add delay between tests to respect rate limits
            await asyncio.sleep(2)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 Demo Results Summary")
        print(f"{'='*60}")
        print(f"✅ Successful Tests: {successful_tests}/{len(demo_queries)}")
        print(f"❌ Failed Tests: {len(demo_queries) - successful_tests}/{len(demo_queries)}")
        
        if successful_tests > 0:
            avg_time = sum(r["processing_time"] for r in results if r["success"]) / successful_tests
            print(f"⏱️  Average Processing Time: {avg_time:.2f}s")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅" if result["success"] else "❌"
            time_str = f"{result.get('processing_time', 0):.2f}s" if result["success"] else "Failed"
            print(f"  {i}. {status} {result['query'][:50]}... ({time_str})")
        
        return results

async def main():
    """Main demo function"""
    
    print("🔍 Perplexity-Style GraphRAG System Demo")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable is required")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Create demo system
    demo = PerplexitySystemDemo()
    
    # Setup system
    print("🚀 Setting up system components...")
    setup_success = await demo.setup_system()
    
    if not setup_success:
        print("❌ System setup failed. Please check your configuration.")
        return
    
    print("✅ System setup completed successfully!")
    
    # Run demo
    try:
        results = await demo.run_demo_suite()
        
        # Save results
        output_file = Path("demo_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "demo_timestamp": datetime.now().isoformat(),
                "results": results,
                "system_info": {
                    "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
                    "data_directory": demo.data_dir,
                    "total_tests": len(results),
                    "successful_tests": sum(1 for r in results if r["success"])
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    # Handle missing dependencies gracefully
    try:
        asyncio.run(main())
    except ModuleNotFoundError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install google-generativeai")
    except Exception as e:
        print(f"❌ Demo failed: {e}")