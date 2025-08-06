#!/usr/bin/env python3
"""
GraphRAG Performance Analysis Runner
Executes comprehensive performance analysis and generates optimization reports
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from performance_benchmarker import GraphRAGPerformanceBenchmarker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly configured"""
    required_env_vars = ['GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set the following environment variables:")
        for var in missing_vars:
            logger.error(f"  export {var}=your_api_key_here")
        return False
    
    return True

def print_performance_summary(results: dict):
    """Print a formatted performance summary"""
    print("\n" + "="*80)
    print("🚀 GRAPHRAG PERFORMANCE ANALYSIS RESULTS")
    print("="*80)
    
    # Basic metrics
    query_perf = results.get('query_performance', {})
    avg_time = query_perf.get('overall_avg_time', 0)
    p95_time = query_perf.get('overall_p95_time', 0)
    target_rate = query_perf.get('target_achievement_rate', 0) * 100
    
    print(f"📊 Query Performance:")
    print(f"   Average Response Time: {avg_time:.2f}s")
    print(f"   95th Percentile Time:  {p95_time:.2f}s")
    print(f"   Target Achievement:    {target_rate:.1f}%")
    print(f"   Queries Meeting <10s:  {query_perf.get('queries_meeting_target', 0)}/{query_perf.get('total_queries_tested', 0)}")
    
    # Component performance
    component_perf = results.get('component_performance', {})
    print(f"\n🔧 Component Performance:")
    
    gemini_api = component_perf.get('gemini_api', {})
    if gemini_api:
        print(f"   Gemini API Avg:        {gemini_api.get('avg_response_time', 0):.2f}s")
        print(f"   Gemini API P95:        {gemini_api.get('p95_response_time', 0):.2f}s")
    
    web_search = component_perf.get('web_search', {})
    if web_search:
        print(f"   Web Search Avg:        {web_search.get('avg_search_time', 0):.2f}s")
        print(f"   Web Search P95:        {web_search.get('p95_search_time', 0):.2f}s")
    
    # Memory performance
    memory_perf = results.get('memory_performance', {})
    if memory_perf:
        print(f"\n🧠 Memory Performance:")
        print(f"   Peak Memory Usage:     {memory_perf.get('peak_memory_mb', 0):.0f} MB")
        print(f"   Memory Growth:         {memory_perf.get('net_memory_growth_mb', 0):.1f} MB")
        print(f"   Potential Leak:        {'Yes' if memory_perf.get('potential_memory_leak', False) else 'No'}")
    
    # Concurrency performance
    concurrency_perf = results.get('concurrency_performance', {})
    if concurrency_perf:
        max_concurrency = concurrency_perf.get('max_successful_concurrency', 1)
        print(f"\n⚡ Concurrency Performance:")
        print(f"   Max Successful Level:  {max_concurrency} concurrent queries")
    
    # Bottleneck analysis
    bottlenecks = results.get('bottleneck_analysis', [])
    critical_bottlenecks = [b for b in bottlenecks if b.get('severity') == 'critical']
    high_bottlenecks = [b for b in bottlenecks if b.get('severity') == 'high']
    
    print(f"\n🚨 Bottleneck Analysis:")
    print(f"   Critical Issues:       {len(critical_bottlenecks)}")
    print(f"   High Priority Issues:  {len(high_bottlenecks)}")
    print(f"   Total Issues Found:    {len(bottlenecks)}")
    
    if critical_bottlenecks:
        print(f"\n   Critical Bottlenecks:")
        for bottleneck in critical_bottlenecks[:3]:  # Show top 3
            print(f"   • {bottleneck.get('component', 'Unknown')}: {bottleneck.get('impact_percent', 0):.1f}% impact")
            print(f"     {bottleneck.get('description', 'No description')}")
    
    # Optimization roadmap
    roadmap = results.get('optimization_roadmap', {})
    expected_improvement = roadmap.get('expected_overall_improvement', 'Unknown')
    print(f"\n🎯 Optimization Potential:")
    print(f"   Expected Improvement:  {expected_improvement}")
    
    immediate_actions = roadmap.get('immediate_actions', [])
    if immediate_actions:
        print(f"   Immediate Actions:     {len(immediate_actions)} critical fixes needed")
    
    short_term = roadmap.get('short_term_optimizations', [])
    if short_term:
        print(f"   Short-term Tasks:      {len(short_term)} optimizations identified")
    
    print("\n" + "="*80)
    print("📈 For detailed analysis, check the generated report files:")
    print("   • analysis-reports/performance_benchmark_*.json")
    print("   • analysis-reports/performance_summary_*.txt")
    print("="*80)

def print_top_recommendations(results: dict):
    """Print top optimization recommendations"""
    roadmap = results.get('optimization_roadmap', {})
    immediate_actions = roadmap.get('immediate_actions', [])
    
    if immediate_actions:
        print("\n🔥 TOP PRIORITY OPTIMIZATIONS:")
        print("-" * 50)
        
        for i, action in enumerate(immediate_actions[:3], 1):
            component = action.get('component', 'Unknown')
            impact = action.get('impact_percent', 0)
            improvement = action.get('estimated_improvement', 'Unknown')
            actions_list = action.get('actions', [])
            
            print(f"{i}. {component} (Impact: {impact:.1f}%)")
            print(f"   Expected Improvement: {improvement}")
            print(f"   Recommended Actions:")
            for j, rec_action in enumerate(actions_list[:2], 1):
                print(f"   {j}) {rec_action}")
            print()

def print_perplexity_readiness(results: dict):
    """Assess and print Perplexity-style system readiness"""
    query_perf = results.get('query_performance', {})
    avg_time = query_perf.get('overall_avg_time', 0)
    target_rate = query_perf.get('target_achievement_rate', 0)
    
    component_perf = results.get('component_performance', {})
    web_search_time = component_perf.get('web_search', {}).get('avg_search_time', 0)
    
    concurrency_perf = results.get('concurrency_performance', {})
    max_concurrency = concurrency_perf.get('max_successful_concurrency', 1)
    
    print("\n🎯 PERPLEXITY-STYLE SYSTEM READINESS:")
    print("-" * 50)
    
    # Response time assessment
    if avg_time <= 10.0:
        response_status = "✅ READY"
    elif avg_time <= 15.0:
        response_status = "⚠️  NEEDS OPTIMIZATION"
    else:
        response_status = "❌ NOT READY"
    
    print(f"Response Time Target (<10s):  {response_status} ({avg_time:.2f}s avg)")
    
    # Success rate assessment
    if target_rate >= 90:
        success_status = "✅ READY"
    elif target_rate >= 70:
        success_status = "⚠️  NEEDS IMPROVEMENT"
    else:
        success_status = "❌ NOT READY"
    
    print(f"Success Rate Target (>90%):   {success_status} ({target_rate:.1f}%)")
    
    # Web search performance
    if web_search_time <= 5.0:
        search_status = "✅ READY"
    elif web_search_time <= 8.0:
        search_status = "⚠️  ACCEPTABLE"
    else:
        search_status = "❌ TOO SLOW"
    
    print(f"Web Search Speed (<5s):       {search_status} ({web_search_time:.2f}s avg)")
    
    # Concurrency readiness
    if max_concurrency >= 8:
        concurrency_status = "✅ READY"
    elif max_concurrency >= 4:
        concurrency_status = "⚠️  LIMITED"
    else:
        concurrency_status = "❌ INSUFFICIENT"
    
    print(f"Concurrency Support (>8):     {concurrency_status} ({max_concurrency} max)")
    
    # Overall readiness assessment
    ready_count = sum([
        avg_time <= 10.0,
        target_rate >= 90,
        web_search_time <= 5.0,
        max_concurrency >= 8
    ])
    
    if ready_count >= 3:
        overall_status = "🟢 PRODUCTION READY"
        recommendation = "System is ready for Perplexity-style deployment with minor optimizations."
    elif ready_count >= 2:
        overall_status = "🟡 NEEDS OPTIMIZATION"
        recommendation = "System requires performance optimizations before production deployment."
    else:
        overall_status = "🔴 NOT PRODUCTION READY"
        recommendation = "System requires significant improvements before deployment."
    
    print(f"\nOverall Readiness:            {overall_status}")
    print(f"Recommendation:               {recommendation}")

async def run_quick_analysis():
    """Run a quick performance analysis with reduced scope"""
    logger.info("Running quick performance analysis...")
    
    # Create a minimal config for quick testing
    quick_config = {
        'test_queries': [
            "What is the current price of Bitcoin?",
            "How is AI impacting healthcare?",
            "What are the latest developments in renewable energy?"
        ],
        'concurrency_levels': [1, 2, 4],
        'iterations_per_test': 2,
        'warm_up_iterations': 1,
        'enable_memory_profiling': False
    }
    
    # Save quick config
    quick_config_path = "quick_benchmark_config.json"
    with open(quick_config_path, 'w') as f:
        json.dump(quick_config, f, indent=2)
    
    benchmarker = GraphRAGPerformanceBenchmarker(quick_config_path)
    results = await benchmarker.run_comprehensive_benchmark()
    
    print_performance_summary(results)
    print_perplexity_readiness(results)
    print_top_recommendations(results)
    
    # Cleanup
    if os.path.exists(quick_config_path):
        os.remove(quick_config_path)
    
    return results

async def run_full_analysis():
    """Run comprehensive performance analysis"""
    logger.info("Running comprehensive performance analysis...")
    
    config_path = "benchmark_config.json"
    if not Path(config_path).exists():
        logger.error(f"Configuration file {config_path} not found")
        return None
    
    benchmarker = GraphRAGPerformanceBenchmarker(config_path)
    results = await benchmarker.run_comprehensive_benchmark()
    
    print_performance_summary(results)
    print_perplexity_readiness(results)
    print_top_recommendations(results)
    
    return results

async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='GraphRAG Performance Analysis')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                      help='Analysis mode: quick (3 queries, fast) or full (comprehensive)')
    parser.add_argument('--config', type=str, default='benchmark_config.json',
                      help='Path to benchmark configuration file')
    parser.add_argument('--check-env', action='store_true',
                      help='Check environment configuration and exit')
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    if args.check_env:
        print("✅ Environment configuration is valid")
        return
    
    try:
        if args.mode == 'quick':
            results = await run_quick_analysis()
        else:
            results = await run_full_analysis()
        
        if results:
            logger.info("Performance analysis completed successfully")
        else:
            logger.error("Performance analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())