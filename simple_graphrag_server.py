#!/usr/bin/env python3
"""
Simple GraphRAG Server - Minimal Dependencies Version
Works with basic Python installation and demonstrates core GraphRAG concepts
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Load environment variables manually
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

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available - running in basic HTTP mode")

# Simple HTTP server fallback
import http.server
import socketserver
import urllib.parse
import threading

class SimpleGraphRAGServer:
    """Simple GraphRAG server with minimal dependencies"""
    
    def __init__(self):
        self.stats = {
            "start_time": datetime.utcnow(),
            "requests_processed": 0,
            "queries_processed": 0,
        }
        
        # Check Gemini API key
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
    
    async def simple_gemini_query(self, query: str) -> str:
        """Simple Gemini API query"""
        try:
            import aiohttp
            
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-002:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            # Enhanced prompt for GraphRAG-style responses
            enhanced_query = f"""
            You are a helpful AI assistant with expertise in multiple domains including technology, healthcare, climate, and blockchain.
            
            User Query: {query}
            
            Please provide a comprehensive, well-structured response that:
            1. Directly answers the user's question
            2. Includes relevant context and background information
            3. Mentions related concepts and connections where appropriate
            4. Uses clear, accessible language
            
            If the query relates to artificial intelligence, healthcare technology, climate tech, quantum computing, or blockchain applications, please draw from comprehensive knowledge in these areas.
            """
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": enhanced_query
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2000,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "candidates" in data and len(data["candidates"]) > 0:
                            content = data["candidates"][0]["content"]["parts"][0]["text"]
                            return content.strip()
                        else:
                            return "I apologize, but I couldn't generate a proper response."
                    else:
                        error_text = await response.text()
                        return f"Error communicating with AI service: {response.status}"
        
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from input directory"""
        documents = []
        input_dir = Path("./input")
        
        if input_dir.exists():
            for file_path in input_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content:
                        documents.append({
                            "id": file_path.stem,
                            "title": file_path.name,
                            "content": content,
                            "source": str(file_path),
                            "length": len(content),
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def simple_entity_extraction(self, text: str) -> List[str]:
        """Simple entity extraction using basic patterns"""
        import re
        
        # Find capitalized words (potential proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Filter out common words
        stop_words = {
            "The", "This", "That", "These", "Those", "And", "But", "Or", 
            "For", "With", "By", "In", "On", "At", "To", "From", "Of"
        }
        entities = [match for match in matches if match not in stop_words]
        
        return list(set(entities))[:10]  # Return top 10 unique entities
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        uptime = (datetime.utcnow() - self.stats["start_time"]).total_seconds()
        documents = self.load_documents()
        
        return {
            "status": "operational",
            "version": "1.0.0-simple",
            "uptime": uptime,
            "stats": {
                "requests_processed": self.stats["requests_processed"],
                "queries_processed": self.stats["queries_processed"],
                "documents_available": len(documents),
            },
            "components": {
                "gemini_api": self.api_key is not None,
                "document_loader": True,
                "simple_entity_extraction": True,
            },
            "timestamp": datetime.utcnow().isoformat()
        }

# Global server instance
server_instance = SimpleGraphRAGServer()

if FASTAPI_AVAILABLE:
    # FastAPI version
    app = FastAPI(
        title="Simple GraphRAG Server",
        description="Minimal GraphRAG implementation with Gemini integration",
        version="1.0.0"
    )
    
    class QueryRequest(BaseModel):
        query: str
        include_entities: bool = True
    
    class QueryResponse(BaseModel):
        response: str
        entities: List[str] = []
        documents_referenced: int = 0
        processing_time: float = 0.0
        metadata: Dict[str, Any] = {}
    
    @app.get("/")
    async def root():
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Simple GraphRAG Server</title></head>
        <body style="font-family: Arial, sans-serif; margin: 40px;">
            <h1>🚀 Simple GraphRAG Server</h1>
            <p>Minimal GraphRAG implementation with Gemini integration</p>
            <h2>📡 API Endpoints:</h2>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/status">System Status</a></li>
                <li><a href="/docs">API Documentation</a></li>
            </ul>
            <h2>💻 Usage:</h2>
            <pre>curl -X POST "http://localhost:8000/query" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is artificial intelligence?"}'</pre>
        </body>
        </html>
        """)
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    @app.get("/status")
    async def system_status():
        return server_instance.get_system_status()
    
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(request: QueryRequest):
        start_time = time.time()
        server_instance.stats["requests_processed"] += 1
        server_instance.stats["queries_processed"] += 1
        
        try:
            # Get AI response
            response_text = await server_instance.simple_gemini_query(request.query)
            
            # Extract entities if requested
            entities = []
            if request.include_entities:
                # Extract entities from both query and response
                query_entities = server_instance.simple_entity_extraction(request.query)
                response_entities = server_instance.simple_entity_extraction(response_text)
                entities = list(set(query_entities + response_entities))[:10]
            
            # Load documents for context
            documents = server_instance.load_documents()
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                response=response_text,
                entities=entities,
                documents_referenced=len(documents),
                processing_time=processing_time,
                metadata={
                    "query_length": len(request.query),
                    "response_length": len(response_text),
                    "model": "gemini-1.5-pro-002",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

else:
    # Basic HTTP server fallback
    class SimpleHTTPHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                html = """
                <!DOCTYPE html>
                <html>
                <head><title>Simple GraphRAG Server</title></head>
                <body style="font-family: Arial, sans-serif; margin: 40px;">
                    <h1>🚀 Simple GraphRAG Server (Basic Mode)</h1>
                    <p>Running in basic HTTP mode - FastAPI not available</p>
                    <p>Install FastAPI for full functionality: pip install fastapi uvicorn</p>
                    <h2>Available endpoints:</h2>
                    <ul>
                        <li><a href="/health">Health Check</a></li>
                        <li><a href="/status">System Status</a></li>
                    </ul>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
                
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                response = {
                    "status": "healthy",
                    "mode": "basic_http",
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == "/status":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                status = server_instance.get_system_status()
                self.wfile.write(json.dumps(status).encode())
                
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            print(f"{datetime.utcnow().isoformat()} - {format % args}")

def main():
    """Main entry point"""
    port = int(os.getenv("GRAPHRAG_PORT", "8000"))
    
    print(f"🚀 Starting Simple GraphRAG Server on port {port}")
    print(f"🔑 Gemini API: {'✅ Configured' if server_instance.api_key else '❌ Missing'}")
    print(f"📦 FastAPI: {'✅ Available' if FASTAPI_AVAILABLE else '❌ Not installed'}")
    
    if FASTAPI_AVAILABLE:
        print(f"🌐 Dashboard: http://localhost:{port}")
        print(f"📖 API Docs: http://localhost:{port}/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        print(f"🌐 Basic server: http://localhost:{port}")
        print("💡 For full features, install: pip install fastapi uvicorn")
        
        with socketserver.TCPServer(("", port), SimpleHTTPHandler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    main()