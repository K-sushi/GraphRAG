#!/usr/bin/env python3
"""
Test LightRAG 0.1.0b6 New Component-Based API
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_lightrag_components():
    """Test LightRAG component-based architecture"""
    print("=== LightRAG 0.1.0b6 Component Test ===")
    
    try:
        # Import core components
        from lightrag.core import (
            Component, Generator, Retriever, Embedder, 
            ModelClient, Sequential, LocalDB, Prompt
        )
        print("[OK] Core components imported successfully")
        
        # Check ModelClient for Gemini integration
        print("\nAvailable components:")
        print(f"- Component: {Component}")
        print(f"- Generator: {Generator}")  
        print(f"- Retriever: {Retriever}")
        print(f"- Embedder: {Embedder}")
        print(f"- ModelClient: {ModelClient}")
        print(f"- LocalDB: {LocalDB}")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Failed to import components: {e}")
        return False

def test_model_clients():
    """Test available model clients"""
    print("\n=== Testing Model Clients ===")
    
    try:
        from lightrag.components.model_client import GroqAPIClient, OpenAIClient
        print("[OK] Model clients available:")
        print(f"- GroqAPIClient: {GroqAPIClient}")
        print(f"- OpenAIClient: {OpenAIClient}")
        
        # Check if there's a Gemini client
        try:
            from lightrag.components.model_client import GeminiClient
            print(f"- GeminiClient: {GeminiClient}")
        except ImportError:
            print("- GeminiClient: Not available (expected)")
            
        return True
        
    except ImportError as e:
        print(f"[ERROR] Model clients not available: {e}")
        return False

def create_simple_rag_pipeline():
    """Create a simple RAG pipeline using new API"""
    print("\n=== Creating Simple RAG Pipeline ===")
    
    try:
        from lightrag.core import Generator, Sequential, Prompt
        
        # Create a simple generator component
        # Note: This will use mock for now since we need to figure out Gemini integration
        
        print("[INFO] LightRAG 0.1.0b6 uses component-based architecture")
        print("[INFO] Traditional GraphRAG approach not directly available")
        print("[INFO] Need to build custom pipeline with components")
        
        # Create a basic prompt
        system_prompt = Prompt("You are a helpful assistant that answers questions based on provided context.")
        print(f"[OK] Created prompt component: {system_prompt}")
        
        # Create a sequential pipeline
        pipeline = Sequential()
        print(f"[OK] Created sequential pipeline: {pipeline}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create pipeline: {e}")
        return False

def analyze_current_implementation():
    """Analyze what we have vs what we expected"""
    print("\n=== Current Implementation Analysis ===")
    
    print("Expected (from project files):")
    print("- Traditional LightRAG with built-in GraphRAG features")
    print("- Direct knowledge graph construction")
    print("- Built-in vector + graph hybrid search")
    
    print("\nActual (LightRAG 0.1.0b6):")
    print("- Component-based RAG framework")
    print("- Requires custom pipeline construction")
    print("- More flexible but needs more setup")
    
    print("\nNext steps needed:")
    print("1. Build custom GraphRAG pipeline using components")
    print("2. Create Gemini model client integration")
    print("3. Implement knowledge graph functionality")
    print("4. Add vector store integration")
    
    return True

def main():
    """Main test function"""
    print("Testing LightRAG 0.1.0b6 New API")
    print("=" * 50)
    
    # Test core components
    if not test_lightrag_components():
        return
    
    # Test model clients
    test_model_clients()
    
    # Try to create simple pipeline
    create_simple_rag_pipeline()
    
    # Analyze current situation
    analyze_current_implementation()
    
    print("\n=== Summary ===")
    print("[DISCOVERY] LightRAG 0.1.0b6 is a different framework than expected")
    print("[SOLUTION] Need to adapt our GraphRAG implementation approach")
    print("[STATUS] Environment setup complete, ready for custom implementation")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"[READY] Gemini API key configured: {gemini_key[:10]}...")
        print("[NEXT] Build custom GraphRAG using LightRAG components + Gemini")
    else:
        print("[ERROR] No Gemini API key found")

if __name__ == "__main__":
    main()