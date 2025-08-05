#!/usr/bin/env python3
"""
Explore LightRAG package structure
"""

import lightrag
import pkgutil

print("LightRAG version:", lightrag.__version__)
print("LightRAG path:", lightrag.__path__)

print("\nAvailable modules in LightRAG:")
for importer, modname, ispkg in pkgutil.iter_modules(lightrag.__path__, lightrag.__name__ + '.'):
    print(f"  {modname} (package: {ispkg})")

# Try to find the main class
print("\nTrying to find main classes...")

try:
    from lightrag.lightrag import LightRAG
    print("Found LightRAG class in lightrag.lightrag")
except ImportError as e:
    print(f"Not in lightrag.lightrag: {e}")

try:
    from lightrag.core import LightRAG
    print("Found LightRAG class in lightrag.core")
except ImportError as e:
    print(f"Not in lightrag.core: {e}")

try:
    from lightrag.main import LightRAG
    print("Found LightRAG class in lightrag.main")
except ImportError as e:
    print(f"Not in lightrag.main: {e}")

# Check for any class with RAG in the name
try:
    import lightrag.operate
    print("Found lightrag.operate module")
    print("  Contents:", dir(lightrag.operate))
except ImportError as e:
    print(f"No lightrag.operate: {e}")

try:
    import lightrag.utils
    print("Found lightrag.utils module")
    print("  Contents:", dir(lightrag.utils))
except ImportError as e:
    print(f"No lightrag.utils: {e}")