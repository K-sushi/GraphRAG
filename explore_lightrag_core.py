#!/usr/bin/env python3
"""
Explore LightRAG core modules
"""

import lightrag.core

print("LightRAG core contents:")
print(dir(lightrag.core))

print("\nExploring submodules...")

try:
    import lightrag.core.base_component
    print("lightrag.core.base_component found:")
    print("  Contents:", [x for x in dir(lightrag.core.base_component) if not x.startswith('_')])
except ImportError as e:
    print(f"No base_component: {e}")

try:
    import lightrag.core.component
    print("lightrag.core.component found:")
    print("  Contents:", [x for x in dir(lightrag.core.component) if not x.startswith('_')])
except ImportError as e:
    print(f"No component: {e}")

try:
    import lightrag.core.container
    print("lightrag.core.container found:")
    print("  Contents:", [x for x in dir(lightrag.core.container) if not x.startswith('_')])
except ImportError as e:
    print(f"No container: {e}")

try:
    import lightrag.core.db
    print("lightrag.core.db found:")
    print("  Contents:", [x for x in dir(lightrag.core.db) if not x.startswith('_')])
except ImportError as e:
    print(f"No db: {e}")

try:
    import lightrag.core.embedder
    print("lightrag.core.embedder found:")
    print("  Contents:", [x for x in dir(lightrag.core.embedder) if not x.startswith('_')])
except ImportError as e:
    print(f"No embedder: {e}")

try:
    import lightrag.core.generator
    print("lightrag.core.generator found:")
    print("  Contents:", [x for x in dir(lightrag.core.generator) if not x.startswith('_')])
except ImportError as e:
    print(f"No generator: {e}")

try:
    import lightrag.core.retriever
    print("lightrag.core.retriever found:")
    print("  Contents:", [x for x in dir(lightrag.core.retriever) if not x.startswith('_')])
except ImportError as e:
    print(f"No retriever: {e}")

print("\nLooking for main RAG classes...")

# Check all known possible locations
modules_to_check = [
    'lightrag.core.rag',
    'lightrag.core.lightrag',
    'lightrag.core.graph_rag',
    'lightrag.components',
    'lightrag.components.rag'
]

for module_name in modules_to_check:
    try:
        module = __import__(module_name, fromlist=[''])
        contents = [x for x in dir(module) if not x.startswith('_')]
        print(f"{module_name}: {contents}")
        
        # Look for classes with 'RAG' in name
        rag_classes = [x for x in contents if 'RAG' in x or 'Rag' in x]
        if rag_classes:
            print(f"  RAG classes: {rag_classes}")
            
    except ImportError as e:
        print(f"{module_name}: Not available ({e})")