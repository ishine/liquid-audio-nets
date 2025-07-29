#!/usr/bin/env python3
"""Security check script for liquid-audio-nets repository."""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_secrets_in_code() -> List[Tuple[str, str, int]]:
    """Check for potential secrets in code files."""
    issues = []
    
    # Patterns for common secrets
    secret_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
        (r'secret_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret key'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
        (r'-----BEGIN PRIVATE KEY-----', 'Private key'),
        (r'-----BEGIN RSA PRIVATE KEY-----', 'RSA private key'),
    ]
    
    # File extensions to check
    extensions = ['.py', '.rs', '.cpp', '.c', '.h', '.hpp', '.js', '.ts']
    
    for ext in extensions:
        for file_path in Path('.').rglob(f'*{ext}'):
            if any(skip in str(file_path) for skip in ['.git', 'target', 'build', '__pycache__']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        for pattern, description in secret_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                issues.append((str(file_path), description, line_num))
            except (UnicodeDecodeError, PermissionError):
                continue
                
    return issues


def check_unsafe_functions() -> List[Tuple[str, str, int]]:
    """Check for usage of unsafe functions."""
    issues = []
    
    # Unsafe C/C++ functions
    unsafe_c_patterns = [
        (r'\bstrcpy\s*\(', 'Use strncpy or strlcpy instead'),
        (r'\bstrcat\s*\(', 'Use strncat or strlcat instead'),
        (r'\bsprintf\s*\(', 'Use snprintf instead'),
        (r'\bgets\s*\(', 'Use fgets instead'),
        (r'\bscanf\s*\(', 'Use safer input functions'),
    ]
    
    # Check C/C++ files
    for file_path in Path('.').rglob('*.c'):
        if any(skip in str(file_path) for skip in ['.git', 'target', 'build']):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern, description in unsafe_c_patterns:
                        if re.search(pattern, line):
                            issues.append((str(file_path), description, line_num))
        except (UnicodeDecodeError, PermissionError):
            continue
            
    return issues


def check_dependency_security() -> List[str]:
    """Check for known security issues in dependencies."""
    issues = []
    
    # Check for known vulnerable packages (simplified)
    vulnerable_packages = {
        'pillow': ['<8.2.0'],  # Example - adjust based on current vulnerabilities
    }
    
    # Check Python requirements
    req_files = ['requirements.txt', 'pyproject.toml']
    for req_file in req_files:
        if os.path.exists(req_file):
            try:
                with open(req_file, 'r') as f:
                    content = f.read()
                    for pkg, versions in vulnerable_packages.items():
                        if pkg in content:
                            issues.append(f"Potentially vulnerable package {pkg} found in {req_file}")
            except IOError:
                continue
                
    return issues


def main():
    """Run security checks."""
    print("🔒 Running security checks...")
    
    # Check for secrets
    secret_issues = check_secrets_in_code()
    if secret_issues:
        print("\n❌ Potential secrets found:")
        for file_path, description, line_num in secret_issues:
            print(f"  {file_path}:{line_num} - {description}")
    
    # Check for unsafe functions
    unsafe_issues = check_unsafe_functions()
    if unsafe_issues:
        print("\n⚠️  Unsafe functions found:")
        for file_path, description, line_num in unsafe_issues:
            print(f"  {file_path}:{line_num} - {description}")
    
    # Check dependencies
    dep_issues = check_dependency_security()
    if dep_issues:
        print("\n⚠️  Dependency security issues:")
        for issue in dep_issues:
            print(f"  {issue}")
    
    # Summary
    total_issues = len(secret_issues) + len(unsafe_issues) + len(dep_issues)
    if total_issues == 0:
        print("\n✅ No security issues found!")
        sys.exit(0)
    else:
        print(f"\n❌ Found {total_issues} potential security issues")
        sys.exit(1)


if __name__ == "__main__":
    main()