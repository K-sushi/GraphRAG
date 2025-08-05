#!/usr/bin/env python3
"""
Test basic Gemini functionality without web search tools
"""

import asyncio
import os
import google.generativeai as genai

async def test_basic_gemini():
    """Test basic Gemini API functionality"""
    
    # Configure API
    api_key = 'AIzaSyArtXFCBWzNkK1drm4zS6XjY3L6L2WnAzY'
    genai.configure(api_key=api_key)
    
    print("=== Basic Gemini Test ===")
    
    # Test 1: Basic query without tools
    print("\n--- Test 1: Basic Query (No Tools) ---")
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("What is Bitcoin? Give me a brief explanation.")
        print("SUCCESS: Basic query successful")
        print(f"Response: {response.text[:200]}...")
    except Exception as e:
        print(f"FAILED: Basic query failed: {e}")
    
    # Test 2: Try google_search_retrieval with current library
    print("\n--- Test 2: Web Search Tool Test ---")
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp', tools='google_search_retrieval')
        response = model.generate_content("What is the current Bitcoin price?")
        print("SUCCESS: Web search query successful")
        print(f"Response: {response.text[:200]}...")
        
        # Check if grounding metadata is present
        if hasattr(response, 'grounding_metadata') and response.grounding_metadata:
            print(f"GROUNDING: Sources found: {len(response.grounding_metadata.grounding_chunks)}")
        else:
            print("WARNING: No grounding metadata found")
            
    except Exception as e:
        print(f"FAILED: Web search query failed: {e}")
    
    # Test 3: Try with different models
    print("\n--- Test 3: Different Model Test ---")
    for model_name in ['gemini-1.5-pro', 'gemini-1.5-flash']:
        try:
            model = genai.GenerativeModel(model_name, tools='google_search_retrieval')
            response = model.generate_content("What is the current time?")
            print(f"SUCCESS: Model {model_name} with tools successful")
            break
        except Exception as e:
            print(f"FAILED: Model {model_name} failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_basic_gemini())