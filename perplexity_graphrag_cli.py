#!/usr/bin/env python3
"""
Perplexity-Style GraphRAG CLI
SuperClaude Wave Orchestration - Phase 3A

Command-line interface for the Perplexity-style real-time search + AI reasoning system
"""

import os
import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and check dependencies"""
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is required")
        print("\n📋 Setup Instructions:")
        print("1. Get a Gemini API key from: https://makersuite.google.com/app/apikey")
        print("2. Set the environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("3. Run this script again")
        return False
    
    # Check required modules
    try:
        import google.generativeai as genai
        from gemini_web_search import create_gemini_web_search, create_perplexity_graphrag
        from gemini_llm_provider import create_gemini_llm
        from graphrag_search import create_search_engines
        print("✅ All required modules available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📋 Installation Instructions:")
        print("pip install google-generativeai sentence-transformers faiss-cpu")
        return False

async def interactive_mode(perplexity_system):
    """Interactive chat mode"""
    
    print("\n🔍 Perplexity-Style GraphRAG - Interactive Mode")
    print("=" * 50)
    print("Ask questions about current events, prices, or any topic!")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print("Type 'help' for available commands")
    print()
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("🤔 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation history cleared")
                continue
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip() or f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_conversation(conversation_history, filename)
                continue
            
            # Process query
            print("🔍 Searching and analyzing...")
            
            result = await perplexity_system.process_query(
                user_input, 
                conversation_history=conversation_history[-5:]  # Last 5 exchanges
            )
            
            # Display response
            print(f"\n🤖 Assistant: {result.get('synthesized_response', 'No response available')}")
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": result.get('synthesized_response', '')})
            
            # Show quick stats
            metadata = result.get('processing_metadata', {})
            quality = metadata.get('response_quality', {})
            
            stats = []
            if metadata.get('web_search_performed'):
                stats.append("🌐 Web")
            if metadata.get('graphrag_used'):
                stats.append("🧠 GraphRAG")
            if metadata.get('total_sources', 0) > 0:
                stats.append(f"📚 {metadata['total_sources']} sources")
            if quality.get('overall_score'):
                stats.append(f"⭐ {quality['overall_score']:.1f}/1.0")
            
            if stats:
                print(f"   ({' • '.join(stats)})")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Interactive mode error: {e}")

def print_help():
    """Print help information"""
    print("\n📋 Available Commands:")
    print("  help        - Show this help message")
    print("  quit/exit   - Exit the program")
    print("  clear       - Clear conversation history")
    print("  save [name] - Save conversation to file")
    print("\n💡 Example Queries:")
    print("  • BTC現在価格は？")
    print("  • What's the current Bitcoin price?")
    print("  • Latest AI news today")
    print("  • What is machine learning?")
    print("  • Current weather in Tokyo")
    print()

def save_conversation(history, filename):
    """Save conversation history to file"""
    try:
        output_file = Path(filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "saved_at": datetime.now().isoformat(),
                "conversation_history": history
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Conversation saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save conversation: {e}")

async def single_query_mode(perplexity_system, query, force_web_search=False, show_sources=False):
    """Process a single query and display results"""
    
    print(f"🔍 Query: {query}")
    print("=" * 60)
    
    # Process query
    result = await perplexity_system.process_query(query, force_web_search=force_web_search)
    
    # Display response
    print(f"\n📝 Response:")
    print(result.get('synthesized_response', 'No response available'))
    
    # Display metadata
    metadata = result.get('processing_metadata', {})
    print(f"\n📊 Processing Info:")
    print(f"  • Type: {metadata.get('response_type', 'unknown')}")
    print(f"  • Time: {metadata.get('processing_time_seconds', 0):.2f}s")
    print(f"  • Web Search: {'Yes' if metadata.get('web_search_performed') else 'No'}")
    print(f"  • GraphRAG: {'Yes' if metadata.get('graphrag_used') else 'No'}")
    
    quality = metadata.get('response_quality', {})
    if quality:
        print(f"  • Quality: {quality.get('overall_score', 0):.2f}/1.0 ({quality.get('assessment', 'unknown')})")
        print(f"  • Sources: {quality.get('total_sources', 0)} total")
    
    # Show sources if requested
    if show_sources:
        sources = result.get('all_sources', [])
        if sources:
            print(f"\n📚 Sources ({len(sources)}):")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source.get('title', 'Unknown Title')}")
                if source.get('url'):
                    print(f"     🔗 {source['url']}")
                print(f"     📑 {source.get('source_type', 'unknown')} | {source.get('source_origin', 'unknown')}")
                if source.get('confidence'):
                    print(f"     📊 Confidence: {source['confidence']:.2f}")
                print()

async def demo_mode(perplexity_system):
    """Run demonstration with sample queries"""
    
    demo_queries = [
        "BTC現在価格は？",
        "What is the current Bitcoin price?",
        "Latest AI news today",
        "What is machine learning?",
        "Current weather in Tokyo"
    ]
    
    print("🚀 Running Demo Mode")
    print(f"Testing {len(demo_queries)} sample queries...")
    print()
    
    for i, query in enumerate(demo_queries, 1):
        print(f"📝 Demo {i}/{len(demo_queries)}: {query}")
        print("-" * 40)
        
        try:
            result = await perplexity_system.process_query(query)
            response = result.get('synthesized_response', 'No response')
            
            # Show first 200 characters
            display_response = response[:200] + "..." if len(response) > 200 else response
            print(f"🤖 {display_response}")
            
            metadata = result.get('processing_metadata', {})
            print(f"   ({metadata.get('processing_time_seconds', 0):.1f}s, "
                  f"{'Web+' if metadata.get('web_search_performed') else ''}"
                  f"{'GraphRAG' if metadata.get('graphrag_used') else ''})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
        if i < len(demo_queries):
            await asyncio.sleep(2)  # Rate limiting

async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Perplexity-Style GraphRAG System")
    
    # Operation modes
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Interactive chat mode')
    parser.add_argument('--query', '-q', type=str, 
                       help='Single query to process')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demonstration mode')
    
    # Configuration options
    parser.add_argument('--data-dir', default='./output', 
                       help='GraphRAG data directory')
    parser.add_argument('--force-web', action='store_true', 
                       help='Force web search for all queries')
    parser.add_argument('--show-sources', action='store_true', 
                       help='Show detailed source information')
    parser.add_argument('--config', type=str, 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Check environment
    if not setup_environment():
        return
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load config file: {e}")
    
    # Initialize system
    print("🚀 Initializing Perplexity-Style GraphRAG System...")
    
    try:
        # Import modules
        from gemini_web_search import create_gemini_web_search, create_perplexity_graphrag
        from gemini_llm_provider import create_gemini_llm
        from graphrag_search import create_search_engines
        
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Create components
        web_search_config = config.get("web_search", {
            "search_model": "gemini-2.0-flash-exp",
            "analysis_model": "gemini-1.5-pro-002",
            "synthesis_model": "gemini-1.5-flash-002"
        })
        
        web_search = create_gemini_web_search(api_key, web_search_config)
        
        gemini_provider = create_gemini_llm({
            "api_key": api_key,
            "model": "gemini-2.0-flash-exp"
        })
        
        _, _, hybrid_search = create_search_engines(args.data_dir, gemini_provider)
        
        perplexity_config = config.get("perplexity", {
            "freshness_threshold": 0.7,
            "always_use_graphrag": True
        })
        
        perplexity_system = create_perplexity_graphrag(
            web_search, hybrid_search, perplexity_config
        )
        
        print("✅ System initialized successfully!")
        
        # Run requested mode
        if args.interactive:
            await interactive_mode(perplexity_system)
        elif args.query:
            await single_query_mode(perplexity_system, args.query, args.force_web, args.show_sources)
        elif args.demo:
            await demo_mode(perplexity_system)
        else:
            # Default to interactive mode
            print("No mode specified, starting interactive mode...")
            await interactive_mode(perplexity_system)
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        logger.error(f"System initialization error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")