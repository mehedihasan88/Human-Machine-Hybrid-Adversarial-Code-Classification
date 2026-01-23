#!/usr/bin/env python3"""AST Feature Extractor for Language-Agnostic Code ClassificationThis module provides AST-based feature extraction to handle programming languagesnot seen during training, making the classifier language-agnostic."""import ast
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import re
import warnings
import tokenize
import io
import sys

warnings.filterwarnings("ignore")

class ASTFeatureExtractor:
    """Extract language-agnostic features from code using AST parsing."""
    
    def __init__(self):
        self.feature_cache = {}
        
    def extract_features(self, code: str, language: str = None) -> Dict[str, Any]:
        """
        Extract comprehensive AST-based features from code.
        
        Args:
            code: Source code string
            language: Programming language (optional, for fallback parsing)
            
        Returns:
            Dictionary of AST features
        """
        if not code or not code.strip():
            return self._get_empty_features()
            
        # Use cache for identical code
        code_hash = hash(code)
        if code_hash in self.feature_cache:
            return self.feature_cache[code_hash]
        
        features = {}
        
        # Try Python AST first (most robust)
        if self._is_python_like(code):
            features = self._extract_python_ast_features(code)
        else:
            # Fallback to structural analysis for other languages
            features = self._extract_structural_features(code)
        
        # Add language-agnostic features
        features.update(self._extract_universal_features(code))
        
        # Cache the result
        self.feature_cache[code_hash] = features
        
        return features
    
    def _is_python_like(self, code: str) -> bool:
        """Check if code is Python-like based on syntax patterns."""
        python_indicators = [
            r'def\s+\w+\s*\(',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'print\s*\(',
            r'elif\s+',
            r':\s*$',  # Colon at end of line
            r'#.*$',   # Comments starting with #
        ]
        
        code_lower = code.lower()
        for pattern in python_indicators:
            if re.search(pattern, code_lower, re.MULTILINE):
                return True
        return False
    
    def _extract_python_ast_features(self, code: str) -> Dict[str, Any]:
        """Extract features using Python AST parser."""
        features = {}
        
        try:
            tree = ast.parse(code)
            
            # Node type counts
            node_counts = Counter()
            for node in ast.walk(tree):
                node_counts[type(node).__name__] += 1
            
            # Structural features
            features.update({
                'total_nodes': len(list(ast.walk(tree))),
                'max_depth': self._get_ast_depth(tree),
                'num_functions': node_counts.get('FunctionDef', 0),
                'num_classes': node_counts.get('ClassDef', 0),
                'num_imports': node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0),
                'num_loops': node_counts.get('For', 0) + node_counts.get('While', 0),
                'num_conditionals': node_counts.get('If', 0),
                'num_assignments': node_counts.get('Assign', 0),
                'num_calls': node_counts.get('Call', 0),
                'num_strings': node_counts.get('Constant', 0),
                'num_variables': node_counts.get('Name', 0),
            })
            
            # Control flow complexity
            features['cyclomatic_complexity'] = self._calculate_cyclomatic_complexity(tree)
            
            # Function statistics
            func_stats = self._analyze_functions(tree)
            features.update(func_stats)
            
        except Exception as e:
            # Fallback to structural features if AST parsing fails
            features = self._extract_structural_features(code)
        
        return features
    
    def _extract_structural_features(self, code: str) -> Dict[str, Any]:
        """Extract features from code structure when AST parsing fails."""
        features = {}
        
        # Initialize with default AST values
        features.update({
            'total_nodes': 0,
            'max_depth': 0,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'num_assignments': 0,
            'num_calls': 0,
            'num_strings': 0,
            'num_variables': 0,
            'cyclomatic_complexity': 0,
        })
        
        # Basic structural patterns
        lines = code.split('\n')
        features.update({
            'total_lines': len(lines),
            'non_empty_lines': len([l for l in lines if l.strip()]),
            'avg_line_length': np.mean([len(l) for l in lines]) if lines else 0,
        })
        
        # Pattern matching for common constructs
        patterns = {
            'functions': r'(def\s+\w+\s*\(|function\s+\w+\s*\(|void\s+\w+\s*\(|int\s+\w+\s*\()',
            'classes': r'(class\s+\w+|public\s+class\s+\w+)',
            'loops': r'\b(for|while|do)\s*\(',
            'conditionals': r'\b(if|else\s+if|switch)\s*\(',
            'assignments': r'=',
            'function_calls': r'\w+\s*\(',
            'imports': r'(import|include|using)',
        }
        
        for name, pattern in patterns.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            features[f'num_{name}'] = len(matches)
        
        # Bracket analysis
        features.update(self._analyze_brackets(code))
        
        # Token analysis
        features.update(self._analyze_tokens(code))
        
        return features
    
    def _extract_universal_features(self, code: str) -> Dict[str, Any]:
        """Extract language-agnostic features that work for any programming language."""
        features = {}
        
        # Character-level statistics
        features.update({
            'code_length': len(code),
            'num_chars': len(code),
            'num_alphabetic': sum(c.isalpha() for c in code),
            'num_numeric': sum(c.isdigit() for c in code),
            'num_whitespace': sum(c.isspace() for c in code),
            'num_punctuation': sum(not c.isalnum() and not c.isspace() for c in code),
            'alphabetic_ratio': sum(c.isalpha() for c in code) / max(len(code), 1),
            'numeric_ratio': sum(c.isdigit() for c in code) / max(len(code), 1),
        })
        
        # Word-level statistics
        words = re.findall(r'\b\w+\b', code)
        if words:
            features.update({
                'num_words': len(words),
                'unique_words': len(set(words)),
                'avg_word_length': np.mean([len(w) for w in words]),
                'max_word_length': max(len(w) for w in words),
                'vocabulary_richness': len(set(words)) / len(words),
            })
        else:
            features.update({
                'num_words': 0,
                'unique_words': 0,
                'avg_word_length': 0,
                'max_word_length': 0,
                'vocabulary_richness': 0,
            })
        
        # Line-level statistics
        lines = code.split('\n')
        if lines:
            features.update({
                'num_lines': len(lines),
                'max_line_length': max(len(line) for line in lines),
                'min_line_length': min(len(line) for line in lines),
                'avg_line_length': np.mean([len(line) for line in lines]),
            })
        
        return features
    
    def _get_ast_depth(self, node, current_depth: int = 0) -> int:
        """Calculate maximum depth of AST."""
        if not hasattr(node, 'body') and not hasattr(node, 'orelse'):
            return current_depth
        
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_ast_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _calculate_cyclomatic_complexity(self, tree) -> int:
        """Calculate cyclomatic complexity from AST."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _analyze_functions(self, tree) -> Dict[str, Any]:
        """Analyze function definitions in AST."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return {
                'num_functions': 0,
                'avg_function_length': 0,
                'max_function_length': 0,
                'avg_function_params': 0,
                'max_function_params': 0,
            }
        
        func_lengths = []
        func_params = []
        
        for func in functions:
            # Function length (number of nodes in function body)
            func_nodes = len(list(ast.walk(func)))
            func_lengths.append(func_nodes)
            
            # Number of parameters
            num_params = len(func.args.args)
            func_params.append(num_params)
        
        return {
            'num_functions': len(functions),
            'avg_function_length': np.mean(func_lengths),
            'max_function_length': max(func_lengths),
            'avg_function_params': np.mean(func_params),
            'max_function_params': max(func_params),
        }
    
    def _analyze_brackets(self, code: str) -> Dict[str, Any]:
        """Analyze bracket usage and balance."""
        brackets = {'(': ')', '[': ']', '{': '}', '<': '>'}
        features = {}
        
        for open_bracket, close_bracket in brackets.items():
            open_count = code.count(open_bracket)
            close_count = code.count(close_bracket)
            
            features[f'num_{open_bracket}'] = open_count
            features[f'num_{close_bracket}'] = close_count
            features[f'bracket_balance_{open_bracket}'] = open_count - close_count
        
        return features
    
    def _analyze_tokens(self, code: str) -> Dict[str, Any]:
        """Analyze code tokens using simple tokenization."""
        try:
            # Try Python tokenization first
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            
            token_types = Counter()
            token_lengths = []
            
            for token in tokens:
                if token.type not in (tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                    token_types[tokenize.tok_name[token.type]] += 1
                    if token.string:
                        token_lengths.append(len(token.string))
            
            features = {
                'total_tokens': len(token_lengths),
                'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
                'max_token_length': max(token_lengths) if token_lengths else 0,
            }
            
            # Add token type counts
            for tok_type, count in token_types.items():
                features[f'token_{tok_type.lower()}'] = count
                
        except:
            # Fallback to simple tokenization
            tokens = re.findall(r'\w+|[^\w\s]', code)
            features = {
                'total_tokens': len(tokens),
                'avg_token_length': np.mean([len(t) for t in tokens]) if tokens else 0,
                'max_token_length': max([len(t) for t in tokens]) if tokens else 0,
            }
        
        return features
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary for empty code."""
        return {
            'total_nodes': 0,
            'max_depth': 0,
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'num_assignments': 0,
            'num_calls': 0,
            'num_strings': 0,
            'num_variables': 0,
            'cyclomatic_complexity': 0,
            'code_length': 0,
            'num_chars': 0,
            'num_alphabetic': 0,
            'num_numeric': 0,
            'num_whitespace': 0,
            'num_punctuation': 0,
            'alphabetic_ratio': 0,
            'numeric_ratio': 0,
            'num_words': 0,
            'unique_words': 0,
            'avg_word_length': 0,
            'max_word_length': 0,
            'vocabulary_richness': 0,
            'num_lines': 0,
            'max_line_length': 0,
            'min_line_length': 0,
            'avg_line_length': 0,
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        return list(self._get_empty_features().keys())
