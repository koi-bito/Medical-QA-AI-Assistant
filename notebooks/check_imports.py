import sys

print("Python version:", sys.version)

try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import peft
    print(f"  peft:         {peft.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import trl
    print(f"  trl:          {trl.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import datasets
    print(f"  datasets:     {datasets.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import accelerate
    print(f"  accelerate:   {accelerate.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import chromadb
    print(f"  chromadb:     {chromadb.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import sentence_transformers
    print(f"  sentence-transformers: {sentence_transformers.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import mlflow
    print(f"  mlflow:       {mlflow.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import fastapi
    print(f"  fastapi:      {fastapi.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import gradio
    print(f"  gradio:       {gradio.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import pandas
    print(f"  pandas:       {pandas.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import numpy
    print(f"  numpy:        {numpy.__version__}")
except ImportError as e:
    print(f"  FAILED: {e}")

try:
    import bitsandbytes
    print(f"  bitsandbytes: {bitsandbytes.__version__}")
except ImportError as e:
    print(f"  FAILED (bitsandbytes): {e}")

print("\nAll checks complete!")
