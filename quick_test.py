import asyncio
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyArtXFCBWzNkK1drm4zS6XjY3L6L2WnAzY'

from gemini_web_search import create_gemini_web_search

async def test_system():
    provider = create_gemini_web_search('AIzaSyArtXFCBWzNkK1drm4zS6XjY3L6L2WnAzY')
    
    # Test 1: Freshness analysis
    print("=== Test 1: Freshness Analysis ===")
    result = await provider.analyze_query_freshness('What is Bitcoin?')
    print(f"Query: 'What is Bitcoin?'")
    print(f"Requires web search: {result['requires_web_search']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Search type: {result['search_type']}")
    
    # Test 2: Time-sensitive query
    print("\n=== Test 2: Time-sensitive Query ===")
    result2 = await provider.analyze_query_freshness('BTC current price today')
    print(f"Query: 'BTC current price today'")
    print(f"Requires web search: {result2['requires_web_search']}")
    print(f"Confidence: {result2['confidence']}")
    print(f"Search type: {result2['search_type']}")
    
    # Test 3: Actual web search (if grounding works)
    print("\n=== Test 3: Web Search Attempt ===")
    try:
        search_result = await provider.perform_web_search('current time')
        print(f"Web search successful")
        print(f"Response length: {len(search_result.get('response', ''))}")
        print(f"Sources found: {len(search_result.get('sources', []))}")
        print(f"First 100 chars: {search_result.get('response', '')[:100]}...")
    except Exception as e:
        print(f"Web search failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_system())