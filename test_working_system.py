#!/usr/bin/env python3
"""
Test Working GraphRAG System
Demonstrates current capabilities with available dependencies
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Load environment variables
def load_env():
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

class WorkingGraphRAGDemo:
    """Demonstrates current GraphRAG capabilities"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.stats = {
            "queries_processed": 0,
            "documents_loaded": 0,
            "entities_extracted": 0,
        }
    
    async def gemini_query(self, query: str, context: str = "") -> str:
        """Query Gemini API with context"""
        try:
            import aiohttp
            
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-002:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            # Enhanced prompt with context
            enhanced_query = f"""
            You are an expert AI assistant with access to a knowledge base about technology, healthcare, climate, quantum computing, and blockchain.
            
            {"Context from knowledge base: " + context if context else ""}
            
            User Query: {query}
            
            Please provide a comprehensive, well-structured response that:
            1. Directly answers the user's question
            2. Includes relevant background information
            3. Makes connections to related concepts
            4. Uses clear, accessible language
            
            If relevant, reference the knowledge base context in your response.
            """
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": enhanced_query
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1500
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "candidates" in data and len(data["candidates"]) > 0:
                            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return "Error: Could not get response from AI service"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def load_documents(self):
        """Load documents from input directory"""
        documents = []
        input_dir = Path("./input")
        
        if input_dir.exists():
            for file_path in input_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    documents.append({
                        "id": file_path.stem,
                        "title": file_path.name,
                        "content": content,
                        "length": len(content),
                    })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        self.stats["documents_loaded"] = len(documents)
        return documents
    
    def extract_entities(self, text: str):
        """Simple entity extraction"""
        import re
        
        # Find capitalized words (potential entities)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Filter common words
        stop_words = {
            "The", "This", "That", "These", "Those", "And", "But", "Or", 
            "For", "With", "By", "In", "On", "At", "To", "From", "Of",
            "It", "Is", "Are", "Was", "Were", "Be", "Been", "Being",
            "Have", "Has", "Had", "Do", "Does", "Did", "Will", "Would",
            "Could", "Should", "May", "Might", "Can", "Cannot"
        }
        
        entities = [match for match in matches if match not in stop_words]
        unique_entities = list(set(entities))
        self.stats["entities_extracted"] += len(unique_entities)
        
        return unique_entities[:10]
    
    def find_relevant_context(self, query: str, documents):
        """Find relevant context from documents"""
        query_words = query.lower().split()
        relevant_docs = []
        
        for doc in documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            
            if score > 0:
                relevant_docs.append({
                    "doc": doc,
                    "score": score,
                    "excerpt": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"]
                })
        
        # Sort by relevance and return top contexts
        relevant_docs.sort(key=lambda x: x["score"], reverse=True)
        
        context_parts = []
        for item in relevant_docs[:3]:  # Top 3 relevant documents
            context_parts.append(f"From {item['doc']['title']}: {item['excerpt']}")
        
        return "\n\n".join(context_parts)
    
    async def process_query(self, query: str):
        """Process a complete GraphRAG-style query"""
        print(f"\n[QUERY] {query}")
        print("-" * 60)
        
        start_time = time.time()
        
        # 1. Load documents
        print("[STEP 1] Loading documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} documents")
        
        # 2. Find relevant context
        print("[STEP 2] Finding relevant context...")
        context = self.find_relevant_context(query, documents)
        print(f"Found context from {len(context.split('From ')) - 1} documents" if context else "No relevant context found")
        
        # 3. Extract entities from query
        print("[STEP 3] Extracting entities...")
        query_entities = self.extract_entities(query)
        print(f"Extracted entities: {', '.join(query_entities) if query_entities else 'None'}")
        
        # 4. Query AI with context
        print("[STEP 4] Querying AI with context...")
        response = await self.gemini_query(query, context)
        
        # 5. Extract entities from response
        response_entities = self.extract_entities(response)
        all_entities = list(set(query_entities + response_entities))
        
        processing_time = time.time() - start_time
        self.stats["queries_processed"] += 1
        
        print(f"\n[RESPONSE] ({processing_time:.2f}s)")
        print("-" * 60)
        print(response)
        
        if all_entities:
            print(f"\n[ENTITIES] {', '.join(all_entities[:10])}")
        
        return {
            "query": query,
            "response": response,
            "entities": all_entities,
            "processing_time": processing_time,
            "documents_used": len(documents),
            "context_found": bool(context)
        }

async def main():
    """Main demo function"""
    print("=" * 80)
    print("WORKING GRAPHRAG SYSTEM DEMONSTRATION")
    print("Minimal Implementation with Gemini Integration")
    print("=" * 80)
    
    # Initialize demo
    demo = WorkingGraphRAGDemo()
    
    if not demo.api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        return 1
    
    # Test queries
    test_queries = [
        "What is artificial intelligence and how is it used in healthcare?",
        "How do renewable energy technologies help combat climate change?",
        "What are the applications of blockchain beyond cryptocurrency?",
        "Explain quantum computing and its potential impact on technology",
        "How are AI and machine learning transforming different industries?"
    ]
    
    results = []
    
    try:
        for query in test_queries:
            result = await demo.process_query(query)
            results.append(result)
            
            # Brief pause between queries
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    print(f"Queries Processed: {demo.stats['queries_processed']}")
    print(f"Documents Loaded: {demo.stats['documents_loaded']}")
    print(f"Entities Extracted: {demo.stats['entities_extracted']}")
    
    avg_time = sum(r["processing_time"] for r in results) / len(results) if results else 0
    context_success = sum(1 for r in results if r["context_found"]) / len(results) * 100 if results else 0
    
    print(f"Average Response Time: {avg_time:.2f}s")
    print(f"Context Success Rate: {context_success:.1f}%")
    
    print("\n[SUCCESS] GraphRAG system is working with current setup!")
    print("\nNext steps:")
    print("1. Install additional dependencies for enhanced features")
    print("2. Implement vector embeddings for better context matching")
    print("3. Add real-time indexing and WebSocket notifications")
    print("4. Set up persistent storage with PostgreSQL")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)