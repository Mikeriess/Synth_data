import os
import sys

"""
Fine-tune Gemma using Axolotl.

Installation:
    pip install -U axolotl[flash-attn] bitsandbytes

Usage:
    # Train with default config
    python train.py

    # Train with custom config
    python train.py --config path/to/config.json
"""

def main():
      
    # Call axolotl CLI directly
    cmd = "axolotl train train_config.yml"
    #cmd = "axolotl train train_config_8gpu.yml"
    sys.exit(os.system(cmd))

if __name__ == "__main__":
    main() 