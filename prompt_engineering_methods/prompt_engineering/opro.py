import random
import numpy as np
import requests
import json
from abc import ABC, abstractmethod
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import re
import ast
from base_optimizer import BaseCodePromptOptimizer
from ollama_client import OllamaClient
