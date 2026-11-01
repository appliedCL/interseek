"""Versioned checkpoint files with atomic latest/best pointers and verified backups."""
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from .base import sha256

FORMAT = "interseq-full-model-v1"
