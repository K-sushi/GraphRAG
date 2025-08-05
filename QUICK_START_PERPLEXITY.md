# 🔍 Quick Start: Perplexity-Style GraphRAG System

**SuperClaude Wave Orchestration - Phase 3A Complete**

Your GraphRAG system now has **Perplexity-style real-time search + AI reasoning** capabilities! Ask questions like "BTC現在価格は？" and get real-time web search results combined with your GraphRAG knowledge.

## 🚀 What's New in Phase 3A

### ✅ **Perplexity-Style Features Now Available:**

1. **🌐 Real-time Web Search** - Live information retrieval using Gemini API
2. **🧠 Intelligent Query Analysis** - Automatic detection of time-sensitive queries  
3. **📊 Source Integration** - Combines web results with existing GraphRAG knowledge
4. **⚡ Smart Synthesis** - AI-powered response generation with source attribution
5. **💬 Interactive Chat** - Natural conversation interface with context memory
6. **📱 CLI Interface** - Easy-to-use command-line tool

### 🎯 **Perfect For Queries Like:**

- **Financial**: "BTC現在価格は？", "Current Bitcoin price", "Stock market today"
- **News**: "Latest AI news", "Breaking news today", "Current events"
- **Weather**: "Weather in Tokyo now", "Current temperature"
- **General**: "What is machine learning?", "Explain quantum computing"

## 🛠️ Setup (5 minutes)

### 1. Get Gemini API Key

```bash
# Get your free Gemini API key from:
# https://makersuite.google.com/app/apikey

# Set environment variable
export GEMINI_API_KEY='your-api-key-here'
```

### 2. Install Dependencies

```bash
# Install required packages
pip install google-generativeai sentence-transformers faiss-cpu

# Verify installation
python -c "import google.generativeai; print('✅ Gemini API ready')"
```

### 3. Test System

```bash
# Quick test
python perplexity_graphrag_cli.py --demo

# Interactive mode
python perplexity_graphrag_cli.py --interactive
```

## 💬 Interactive Mode Usage

### Start Interactive Chat

```bash
python perplexity_graphrag_cli.py --interactive
```

### Example Conversation

```
🔍 Perplexity-Style GraphRAG - Interactive Mode
==================================================
Ask questions about current events, prices, or any topic!

🤔 You: BTC現在価格は？

🔍 Searching and analyzing...

🤖 Assistant: 現在のビットコイン（BTC）価格は約 $43,250 USD です。

過去24時間で約2.5%上昇しています。主要取引所での価格：
- Coinbase: $43,240
- Binance: $43,260  
- Kraken: $43,235

この価格上昇は、機関投資家の関心増加と規制環境の改善によるものとされています。

   (🌐 Web • 🧠 GraphRAG • 📚 5 sources • ⭐ 0.9/1.0)

🤔 You: What factors are driving this price increase?

🔍 Searching and analyzing...

🤖 Assistant: Several key factors are driving Bitcoin's recent price increase:

1. **Institutional Adoption**: Major companies and investment funds continue to add Bitcoin to their portfolios
2. **Regulatory Clarity**: Improved regulatory frameworks in key markets
3. **ETF Speculation**: Ongoing discussions about Bitcoin ETF approvals
4. **Macroeconomic Factors**: Inflation concerns and currency devaluation fears
5. **Technical Analysis**: Breaking above key resistance levels around $42,000

Market analysts suggest this could be the beginning of a longer-term bullish trend, though volatility remains high.

   (🌐 Web • 🧠 GraphRAG • 📚 7 sources • ⭐ 0.8/1.0)
```

### Available Commands

- `help` - Show available commands
- `quit` / `exit` - Exit the program  
- `clear` - Clear conversation history
- `save [filename]` - Save conversation to file

## 🔧 Single Query Mode

### Test Specific Queries

```bash
# Test crypto prices
python perplexity_graphrag_cli.py --query "BTC現在価格は？" --show-sources

# Test current events
python perplexity_graphrag_cli.py --query "Latest AI news today" --show-sources

# Force web search for any query
python perplexity_graphrag_cli.py --query "What is Python?" --force-web
```

### Example Output

```bash
$ python perplexity_graphrag_cli.py --query "Current Bitcoin price" --show-sources

🔍 Query: Current Bitcoin price
============================================================

📝 Response:
The current Bitcoin price is approximately $43,250 USD, showing a 2.5% increase over the past 24 hours. This represents a continuation of the bullish momentum that began earlier this week...

📊 Processing Info:
  • Type: web_graphrag_synthesis
  • Time: 3.2s
  • Web Search: Yes
  • GraphRAG: Yes
  • Quality: 0.9/1.0 (high)
  • Sources: 5 total

📚 Sources (5):
  1. CoinGecko - Bitcoin Price Chart
     🔗 https://www.coingecko.com/en/coins/bitcoin
     📑 web_search | web_search
     📊 Confidence: 0.92

  2. CoinMarketCap - BTC/USD
     🔗 https://coinmarketcap.com/currencies/bitcoin/
     📑 web_search | web_search  
     📊 Confidence: 0.90

  3. Knowledge Graph: Cryptocurrency Analysis
     📑 knowledge_graph_text | graphrag
     📊 Confidence: 0.85
```

## 🚀 Demo Mode

### Run All Sample Queries

```bash
python perplexity_graphrag_cli.py --demo
```

### Sample Output

```
🚀 Running Demo Mode
Testing 5 sample queries...

📝 Demo 1/5: BTC現在価格は？
----------------------------------------
🤖 現在のビットコイン価格は約$43,250 USDで、過去24時間で2.5%上昇しています...
   (3.1s, Web+GraphRAG)

📝 Demo 2/5: What is the current Bitcoin price?
----------------------------------------
🤖 Bitcoin is currently trading at approximately $43,250 USD, representing a 2.5% gain...
   (2.8s, Web+GraphRAG)

📝 Demo 3/5: Latest AI news today
----------------------------------------
🤖 Today's major AI developments include OpenAI's latest model updates, Google's breakthrough...
   (4.2s, Web+GraphRAG)

📝 Demo 4/5: What is machine learning?
----------------------------------------
🤖 Machine learning is a subset of artificial intelligence that enables computers to learn...
   (2.1s, GraphRAG)

📝 Demo 5/5: Current weather in Tokyo
----------------------------------------
🤖 Tokyo is currently experiencing partly cloudy conditions with a temperature of 18°C...
   (3.5s, Web+GraphRAG)
```

## ⚙️ Advanced Configuration

### Create Config File

```json
{
  "web_search": {
    "search_model": "gemini-2.0-flash-exp",
    "analysis_model": "gemini-1.5-pro-002", 
    "synthesis_model": "gemini-1.5-flash-002",
    "max_search_results": 10,
    "cache_duration": 300
  },
  "perplexity": {
    "freshness_threshold": 0.7,
    "always_use_graphrag": true,
    "max_response_time": 30
  }
}
```

### Use Config File

```bash
python perplexity_graphrag_cli.py --config config.json --interactive
```

## 🔍 How It Works

### 1. **Query Analysis**
- Detects time-sensitive keywords (current, now, today, price)
- Analyzes query freshness requirements
- Determines optimal search strategy

### 2. **Intelligent Search**
- **Web Search**: Real-time information via Gemini grounding
- **GraphRAG**: Existing knowledge graph analysis
- **Parallel Processing**: Both sources processed simultaneously

### 3. **Smart Synthesis**
- Combines web results with GraphRAG insights
- Prioritizes current information for time-sensitive queries
- Provides source attribution and confidence scores

### 4. **Quality Assessment**
- Response completeness and accuracy
- Source diversity and reliability
- Information freshness and relevance

## 🧪 Testing Your System

### Test Different Query Types

```bash
# Time-sensitive queries (should use web search)
"BTC price now"
"Current weather Tokyo"
"Latest news today"
"Stock market today"

# General knowledge (may use GraphRAG only)
"What is Python programming?"
"Explain machine learning"
"History of computers"

# Mixed queries (should use both)
"Recent developments in AI"
"Current cryptocurrency trends"
"Today's technology news"
```

### Check System Health

```bash
# Test system components
python test_perplexity_system.py

# Run comprehensive demo
python test_perplexity_system.py --verbose
```

## 🎯 Success Criteria

Your Perplexity-style system is working correctly if:

✅ **Real-time queries** get current information from web search  
✅ **General queries** use GraphRAG knowledge appropriately  
✅ **Response time** is under 10 seconds for most queries  
✅ **Source attribution** is provided with confidence scores  
✅ **Quality scores** are consistently above 0.7/1.0  

## 🔧 Troubleshooting

### Common Issues

**❌ "GEMINI_API_KEY not found"**
```bash
# Solution: Set your API key
export GEMINI_API_KEY='your-key-here'
```

**❌ "Module not found: google.generativeai"**
```bash
# Solution: Install dependencies  
pip install google-generativeai
```

**❌ "GraphRAG data not found"**
```bash
# Solution: Ensure GraphRAG data exists
ls -la ./output/
# Should show: entities.json, relationships.json, etc.
```

**❌ "Rate limit exceeded"**
```bash
# Solution: System has built-in rate limiting
# Wait a few seconds between queries
```

### Performance Tips

- **Use caching**: Results are cached for 5 minutes by default
- **Batch queries**: Use demo mode for multiple tests
- **Monitor API usage**: Check Gemini API quotas
- **Optimize GraphRAG data**: Ensure indexed data is relevant

## 🎉 What's Next?

### Phase 3B - Coming Soon

- **Streamlit Chat UI**: Web-based interface
- **Performance Optimization**: Sub-5-second responses  
- **PostgreSQL Integration**: Persistent storage
- **Advanced Analytics**: Usage metrics and insights

### Try Advanced Features

```bash
# Save conversations
python perplexity_graphrag_cli.py -i
> save my_conversation.json

# Test with conversation history
python perplexity_graphrag_cli.py -q "What did we discuss about Bitcoin?"

# Performance testing
python test_perplexity_system.py --benchmark
```

---

🎉 **Congratulations!** Your GraphRAG system now has Perplexity-style capabilities. Ask any question and get intelligent, source-attributed responses combining real-time web search with your knowledge graph!

**Need help?** Check the logs in `./cache/` or run with `--verbose` for detailed debugging information.