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
# -----------------------------
# Code Quality Evaluator
# -----------------------------
class CodeQualityEvaluator:
    """Evaluate code quality with multiple metrics"""
    
    @staticmethod
    def extract_code_block(text: str) -> str:
        """Extract code from generated text"""
        # Look for code blocks
        code_block_patterns = [
            r'```(?:python)?\s*(.*?)```',
            r'```\s*(.*?)```',
            r'def\s+.*?(?=\n\n|\n#|\nclass|\ndef|\Z)',
            r'class\s+.*?(?=\n\n|\n#|\nclass|\ndef|\Z)'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no code blocks found, try to extract Python-like content
        lines = text.split('\n')
        code_lines = []
        for line in lines:
            if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ', 'return', '    ']):
                code_lines.append(line)
            elif code_lines and line.strip():  # Continue if we're already in code
                code_lines.append(line)
            elif code_lines and not line.strip():  # Empty line in code
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    @staticmethod
    def syntax_check(code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True, "Valid syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    @staticmethod
    def functional_test(code: str, test_cases: List[Dict]) -> float:
        """Test if code works with provided test cases"""
        if not code.strip():
            return 0.0
        
        try:
            # Create a safe execution environment
            exec_globals = {'__builtins__': __builtins__}
            exec(code, exec_globals)
            
            passed_tests = 0
            total_tests = len(test_cases)
            
            for test in test_cases:
                try:
                    func_name = test['function']
                    inputs = test['input']
                    expected = test['expected']
                    
                    if func_name in exec_globals:
                        if isinstance(inputs, list):
                            result = exec_globals[func_name](*inputs)
                        else:
                            result = exec_globals[func_name](inputs)
                        
                        if result == expected:
                            passed_tests += 1
                except Exception:
                    continue
            
            return passed_tests / total_tests if total_tests > 0 else 0.0
            
        except Exception:
            return 0.0
    
    @staticmethod
    def code_quality_metrics(code: str) -> Dict[str, float]:
        """Calculate various code quality metrics"""
        if not code.strip():
            return {
                'readability': 0.0,
                'complexity': 1.0,
                'documentation': 0.0,
                'best_practices': 0.0
            }
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Readability (based on naming, spacing, structure)
        readability_score = 0.0
        readability_factors = 0
        
        # Check for meaningful variable names
        if re.search(r'\b[a-z_][a-z_0-9]{2,}\b', code):
            readability_score += 0.25
        readability_factors += 1
        
        # Check for proper indentation
        indented_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        if indented_lines and len(indented_lines) / len(non_empty_lines) > 0.1:
            readability_score += 0.25
        readability_factors += 1
        
        # Check for blank lines (structure)
        if '' in lines:
            readability_score += 0.25
        readability_factors += 1
        
        # Check for reasonable line length
        long_lines = [line for line in lines if len(line) > 100]
        if len(long_lines) / max(len(non_empty_lines), 1) < 0.2:
            readability_score += 0.25
        readability_factors += 1
        
        readability = readability_score
        
        # Complexity (inverse of cyclomatic complexity indicators)
        complexity_indicators = len(re.findall(r'\b(if|for|while|elif|except|and|or)\b', code))
        complexity = max(0.1, 1.0 - (complexity_indicators * 0.1))
        
        # Documentation (comments and docstrings)
        doc_score = 0.0
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
        
        if comment_lines:
            doc_score += 0.5
        if docstring_count > 0:
            doc_score += 0.5
            
        # Best practices
        practices_score = 0.0
        practices_count = 0
        
        # Function definitions
        if 'def ' in code:
            practices_score += 0.25
        practices_count += 1
        
        # Proper imports
        if re.search(r'^(import|from)\s+', code, re.MULTILINE):
            practices_score += 0.25
        practices_count += 1
        
        # Error handling
        if 'try:' in code or 'except' in code:
            practices_score += 0.25
        practices_count += 1
        
        # Return statements
        if 'return' in code:
            practices_score += 0.25
        practices_count += 1
        
        best_practices = practices_score
        
        return {
            'readability': readability,
            'complexity': complexity,
            'documentation': doc_score,
            'best_practices': best_practices
        }
