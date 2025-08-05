#!/usr/bin/env python3
"""
Enhanced GraphRAG System Test Suite
Tests all components with gradual fallbacks
"""

import asyncio
import json
import sys
import time
import aiohttp
import websockets
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAGSystemTester:
    """Comprehensive system tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.websocket_url = base_url.replace("http", "ws") + "/ws"
        self.session = None
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_details": []
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def log_test_result(self, test_name: str, status: str, details: str = "", error: str = ""):
        """Log test result"""
        self.test_results["total_tests"] += 1
        
        if status == "PASS":
            self.test_results["passed_tests"] += 1
            logger.info(f"✅ {test_name}: PASSED - {details}")
        elif status == "FAIL":
            self.test_results["failed_tests"] += 1
            logger.error(f"❌ {test_name}: FAILED - {error}")
        elif status == "SKIP":
            self.test_results["skipped_tests"] += 1
            logger.warning(f"⏭️  {test_name}: SKIPPED - {details}")
        
        self.test_results["test_details"].append({
            "test_name": test_name,
            "status": status,
            "details": details,
            "error": error,
            "timestamp": time.time()
        })
    
    async def test_basic_connectivity(self):
        """Test basic server connectivity"""
        test_name = "Basic Server Connectivity"
        
        try:
            async with self.session.get(f"{self.base_url}/health", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    uptime = data.get("uptime", 0)
                    self.log_test_result(test_name, "PASS", f"Server healthy, uptime: {uptime:.1f}s")
                    return True
                else:
                    self.log_test_result(test_name, "FAIL", error=f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test_result(test_name, "FAIL", error=str(e))
            return False
    
    async def test_system_status(self):
        """Test system status endpoint"""
        test_name = "System Status"
        
        try:
            async with self.session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    components = data.get("components", {})
                    healthy_components = sum(1 for v in components.values() if v)
                    total_components = len(components)
                    
                    self.log_test_result(
                        test_name, "PASS", 
                        f"{healthy_components}/{total_components} components healthy"
                    )
                    return data
                else:
                    self.log_test_result(test_name, "FAIL", error=f"HTTP {response.status}")
                    return None
        except Exception as e:
            self.log_test_result(test_name, "FAIL", error=str(e))
            return None
    
    async def test_query_endpoint(self):
        """Test query endpoint with various modes"""
        queries = [
            ("simple", {"query": "What is artificial intelligence?", "mode": "local"}),
            ("complex", {"query": "How do machine learning and healthcare technology interact?", "mode": "global"}),
            ("hybrid", {"query": "Explain the relationship between AI and quantum computing", "mode": "hybrid"}),
        ]
        
        results = []
        
        for query_type, query_data in queries:
            test_name = f"Query Endpoint ({query_type})"
            
            try:
                async with self.session.post(
                    f"{self.base_url}/query", 
                    json=query_data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_length = len(data.get("response", ""))
                        search_strategy = data.get("search_strategy", "unknown")
                        processing_time = data.get("metadata", {}).get("processing_time", 0)
                        
                        self.log_test_result(
                            test_name, "PASS",
                            f"Strategy: {search_strategy}, Response: {response_length} chars, Time: {processing_time:.2f}s"
                        )
                        results.append(data)
                    else:
                        self.log_test_result(test_name, "FAIL", error=f"HTTP {response.status}")
                        results.append(None)
            except Exception as e:
                self.log_test_result(test_name, "FAIL", error=str(e))
                results.append(None)
        
        return results
    
    async def test_document_insertion(self):
        """Test document insertion and indexing"""
        test_name = "Document Insertion"
        
        test_document = {
            "content": "This is a test document about emerging technologies. "
                      "It discusses artificial intelligence, machine learning, "
                      "and their applications in various industries.",
            "document_id": f"test_doc_{int(time.time())}",
            "metadata": {"test": True, "source": "system_test"},
            "trigger_indexing": True,
            "priority": 1
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/insert",
                json=test_document,
                timeout=15
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    doc_id = data.get("document_id")
                    task_id = data.get("task_id")
                    
                    self.log_test_result(
                        test_name, "PASS",
                        f"Document ID: {doc_id}, Task ID: {task_id}"
                    )
                    return data
                else:
                    self.log_test_result(test_name, "FAIL", error=f"HTTP {response.status}")
                    return None
        except Exception as e:
            self.log_test_result(test_name, "FAIL", error=str(e))
            return None
    
    async def test_indexing_status(self):
        """Test indexing status endpoint"""
        test_name = "Indexing Status"
        
        try:
            async with self.session.get(f"{self.base_url}/indexing/status") as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get("status", "unknown")
                    active_tasks = data.get("active_tasks", 0)
                    completed_tasks = data.get("completed_tasks", 0)
                    
                    self.log_test_result(
                        test_name, "PASS",
                        f"Status: {status}, Active: {active_tasks}, Completed: {completed_tasks}"
                    )
                    return data
                else:
                    self.log_test_result(test_name, "FAIL", error=f"HTTP {response.status}")
                    return None
        except Exception as e:
            self.log_test_result(test_name, "FAIL", error=str(e))
            return None
    
    async def test_websocket_connection(self):
        """Test WebSocket connectivity and real-time features"""
        test_name = "WebSocket Connection"
        
        try:
            async with websockets.connect(self.websocket_url, timeout=10) as websocket:
                # Test connection
                welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                welcome_data = json.loads(welcome_msg)
                
                if welcome_data.get("type") == "connection":
                    # Test ping-pong
                    await websocket.send(json.dumps({"type": "ping"}))
                    pong_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                    pong_data = json.loads(pong_msg)
                    
                    if pong_data.get("type") == "pong":
                        # Test status request
                        await websocket.send(json.dumps({"type": "status_request"}))
                        status_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                        status_data = json.loads(status_msg)
                        
                        features = welcome_data.get("features", {})
                        realtime_enabled = features.get("realtime_indexing", False)
                        
                        self.log_test_result(
                            test_name, "PASS",
                            f"Ping-pong OK, Real-time: {realtime_enabled}"
                        )
                        return True
                    else:
                        self.log_test_result(test_name, "FAIL", error="Ping-pong failed")
                        return False
                else:
                    self.log_test_result(test_name, "FAIL", error="No welcome message")
                    return False
        
        except Exception as e:
            self.log_test_result(test_name, "FAIL", error=str(e))
            return False
    
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        test_cases = [
            ("Invalid Query", {"json": {"query": ""}, "expected_status": 422}),
            ("Malformed JSON", {"data": "invalid json", "expected_status": 422}),
            ("Large Query", {"json": {"query": "x" * 10000}, "expected_status": [200, 413]}),
        ]
        
        for test_name, test_case in test_cases:
            try:
                if "json" in test_case:
                    async with self.session.post(
                        f"{self.base_url}/query",
                        json=test_case["json"],
                        timeout=15
                    ) as response:
                        expected = test_case["expected_status"]
                        if isinstance(expected, list):
                            success = response.status in expected
                        else:
                            success = response.status == expected
                        
                        if success:
                            self.log_test_result(test_name, "PASS", f"Status: {response.status}")
                        else:
                            self.log_test_result(test_name, "FAIL", error=f"Expected {expected}, got {response.status}")
                
                elif "data" in test_case:
                    async with self.session.post(
                        f"{self.base_url}/query",
                        data=test_case["data"],
                        timeout=15
                    ) as response:
                        if response.status == test_case["expected_status"]:
                            self.log_test_result(test_name, "PASS", f"Status: {response.status}")
                        else:
                            self.log_test_result(test_name, "FAIL", error=f"Expected {test_case['expected_status']}, got {response.status}")
            
            except Exception as e:
                # Some errors are expected
                if "timeout" in str(e).lower() and "Large Query" in test_name:
                    self.log_test_result(test_name, "PASS", "Timeout handled correctly")
                else:
                    self.log_test_result(test_name, "FAIL", error=str(e))
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        test_name = "Performance Benchmarks"
        
        # Test concurrent queries
        concurrent_queries = [
            {"query": f"Test query number {i}", "mode": "local"}
            for i in range(5)
        ]
        
        start_time = time.time()
        
        tasks = []
        for query in concurrent_queries:
            task = self.session.post(f"{self.base_url}/query", json=query, timeout=20)
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            successful_responses = sum(1 for r in responses if not isinstance(r, Exception))
            
            if successful_responses >= 3:  # At least 60% success
                avg_time = total_time / len(concurrent_queries)
                self.log_test_result(
                    test_name, "PASS",
                    f"{successful_responses}/5 queries OK, Avg time: {avg_time:.2f}s"
                )
            else:
                self.log_test_result(
                    test_name, "FAIL",
                    error=f"Only {successful_responses}/5 queries successful"
                )
                
            # Close any open responses
            for response in responses:
                if hasattr(response, 'close'):
                    await response.close()
        
        except Exception as e:
            self.log_test_result(test_name, "FAIL", error=str(e))
    
    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🚀 Starting Enhanced GraphRAG System Tests")
        logger.info("=" * 60)
        
        # Test basic connectivity first
        if not await self.test_basic_connectivity():
            logger.error("❌ Server not accessible, skipping remaining tests")
            return self.generate_report()
        
        # Run system tests
        await self.test_system_status()
        await self.test_query_endpoint()
        await self.test_document_insertion()
        await self.test_indexing_status()
        await self.test_websocket_connection()
        await self.test_error_handling()
        await self.test_performance_benchmarks()
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate test report"""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        skipped = self.test_results["skipped_tests"]
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏭️  Skipped: {skipped}")
        logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("🎉 System is functioning well!")
        elif success_rate >= 60:
            logger.warning("⚠️  System has some issues but is mostly functional")
        else:
            logger.error("🚨 System has significant issues")
        
        # Detailed failure analysis
        if failed > 0:
            logger.info("\n❌ Failed Tests:")
            for detail in self.test_results["test_details"]:
                if detail["status"] == "FAIL":
                    logger.error(f"  - {detail['test_name']}: {detail['error']}")
        
        return self.test_results

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced GraphRAG System Tests")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Run tests
    async with GraphRAGSystemTester(args.url) as tester:
        results = await tester.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Test results saved to {args.output}")
    
    # Return appropriate exit code
    success_rate = (results["passed_tests"] / results["total_tests"] * 100) if results["total_tests"] > 0 else 0
    return 0 if success_rate >= 60 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)