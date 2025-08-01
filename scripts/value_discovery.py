#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for liquid-audio-nets

This script implements the continuous value discovery system that identifies,
scores, and prioritizes work items based on WSJF, ICE, and technical debt metrics.
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import argparse


@dataclass
class ValueItem:
    """Represents a discovered work item with scoring."""
    id: str
    title: str
    description: str
    category: str  # security, technical_debt, feature, performance, etc.
    source: str    # gitHistory, staticAnalysis, etc.
    
    # WSJF Components
    user_business_value: float
    time_criticality: float
    risk_reduction: float
    opportunity_enablement: float
    job_size: float
    
    # ICE Components  
    impact: float
    confidence: float
    ease: float
    
    # Technical Debt
    debt_impact: float
    debt_interest: float
    hotspot_multiplier: float
    
    # Composite Scores
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Metadata
    files_affected: List[str] = None
    estimated_hours: float = 0.0
    priority: str = "medium"  # high, medium, low
    created_at: str = ""
    
    def __post_init__(self):
        if self.files_affected is None:
            self.files_affected = []
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class ValueDiscoveryEngine:
    """Main value discovery engine implementing autonomous prioritization."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        """Initialize the value discovery engine."""
        self.config = self._load_config(config_path)
        self.discovered_items: List[ValueItem] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "scoring": {
                "weights": {
                    "wsjf": 0.6,
                    "ice": 0.1, 
                    "technicalDebt": 0.2,
                    "security": 0.1
                },
                "thresholds": {
                    "minScore": 15.0,
                    "maxRisk": 0.7,
                    "securityBoost": 2.0
                }
            },
            "discovery": {
                "sources": ["gitHistory", "staticAnalysis"]
            }
        }
    
    def discover_from_git_history(self) -> List[ValueItem]:
        """Discover items from Git history (TODOs, FIXMEs, etc.)."""
        items = []
        
        # Search for TODO/FIXME comments
        patterns = [
            (r'TODO:?\s*(.+)', 'todo'),
            (r'FIXME:?\s*(.+)', 'fixme'),
            (r'HACK:?\s*(.+)', 'hack'),
            (r'XXX:?\s*(.+)', 'xxx'),
            (r'NOTE:?\s*(.+)', 'note')
        ]
        
        for pattern, item_type in patterns:
            try:
                result = subprocess.run([
                    'git', 'grep', '-n', '-i', pattern
                ], capture_output=True, text=True, cwd='.')
                
                for line in result.stdout.split('\n'):
                    if line.strip():
                        match = re.search(r'([^:]+):(\d+):(.+)', line)
                        if match:
                            file_path, line_num, content = match.groups()
                            
                            # Extract the actual comment
                            comment_match = re.search(pattern, content, re.IGNORECASE)
                            if comment_match:
                                comment = comment_match.group(1).strip()
                                
                                item = self._create_item_from_comment(
                                    comment, item_type, file_path, int(line_num)
                                )
                                items.append(item)
                                
            except subprocess.SubprocessError:
                continue
                
        return items
    
    def _create_item_from_comment(self, comment: str, item_type: str, 
                                  file_path: str, line_num: int) -> ValueItem:
        """Create a ValueItem from a code comment."""
        # Generate unique ID
        item_id = f"{item_type}-{hash(f'{file_path}:{line_num}:{comment}') % 10000:04d}"
        
        # Categorize based on comment content
        category = self._categorize_comment(comment, item_type)
        
        # Score based on type and content
        scores = self._score_comment_item(comment, item_type, file_path)
        
        return ValueItem(
            id=item_id,
            title=f"{item_type.upper()}: {comment[:50]}...",
            description=f"Found in {file_path}:{line_num} - {comment}",
            category=category,
            source="gitHistory",
            files_affected=[file_path],
            **scores
        )
    
    def _categorize_comment(self, comment: str, item_type: str) -> str:
        """Categorize a comment based on its content."""
        comment_lower = comment.lower()
        
        if any(word in comment_lower for word in ['security', 'auth', 'crypto', 'ssl']):
            return 'security'
        elif any(word in comment_lower for word in ['performance', 'slow', 'optimize', 'memory']):
            return 'performance'
        elif any(word in comment_lower for word in ['refactor', 'cleanup', 'debt', 'ugly']):
            return 'technical_debt'
        elif any(word in comment_lower for word in ['test', 'coverage', 'mock']):
            return 'testing'
        elif item_type == 'hack':
            return 'technical_debt'
        else:
            return 'feature'
    
    def _score_comment_item(self, comment: str, item_type: str, 
                           file_path: str) -> Dict[str, float]:
        """Score a comment-based item."""
        # Base scores by type
        type_scores = {
            'todo': {'urgency': 3, 'complexity': 2},
            'fixme': {'urgency': 7, 'complexity': 4},
            'hack': {'urgency': 8, 'complexity': 6},
            'xxx': {'urgency': 9, 'complexity': 7},
            'note': {'urgency': 1, 'complexity': 1}
        }
        
        base = type_scores.get(item_type, {'urgency': 3, 'complexity': 2})
        
        # Adjust based on file importance (core files = higher impact)
        file_multiplier = 1.0
        if any(core in file_path for core in ['src/', 'python/liquid_audio_nets/', 'tests/']):
            file_multiplier = 1.5
        
        return {
            'user_business_value': base['urgency'] * file_multiplier,
            'time_criticality': base['urgency'],
            'risk_reduction': base['urgency'] * 0.8,
            'opportunity_enablement': 2.0,
            'job_size': base['complexity'],
            'impact': base['urgency'],
            'confidence': 7.0,  # Medium confidence for comment-based items
            'ease': 10 - base['complexity'],
            'debt_impact': base['complexity'] * 2,
            'debt_interest': base['urgency'],
            'hotspot_multiplier': file_multiplier,
            'estimated_hours': base['complexity'] * 0.5
        }
    
    def discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static analysis tools."""
        items = []
        
        # Run Rust clippy
        items.extend(self._run_clippy_analysis())
        
        # Run Python ruff
        items.extend(self._run_ruff_analysis())
        
        return items
    
    def _run_clippy_analysis(self) -> List[ValueItem]:
        """Run Rust clippy analysis."""
        items = []
        
        try:
            result = subprocess.run([
                'cargo', 'clippy', '--all-targets', '--all-features', 
                '--message-format=json'
            ], capture_output=True, text=True, cwd='.')
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('reason') == 'compiler-message':
                            message = data.get('message', {})
                            if message and message.get('level') in ['warning', 'error']:
                                item = self._create_clippy_item(message)
                                if item:
                                    items.append(item)
                    except json.JSONDecodeError:
                        continue
                        
        except subprocess.SubprocessError:
            pass
            
        return items
    
    def _create_clippy_item(self, message: Dict[str, Any]) -> Optional[ValueItem]:
        """Create a ValueItem from a clippy message."""
        if not message.get('spans'):
            return None
            
        span = message['spans'][0]
        file_path = span.get('file_name', '')
        
        # Generate unique ID
        item_id = f"clippy-{hash(message.get('message', '') + file_path) % 10000:04d}"
        
        # Determine severity and category
        level = message.get('level', 'warning')
        code = message.get('code', {}).get('code', '')
        
        category = 'code_quality'
        if 'performance' in code.lower():
            category = 'performance'
        elif 'security' in code.lower():
            category = 'security'
        
        # Score based on severity
        severity_scores = {
            'error': {'urgency': 9, 'complexity': 5},
            'warning': {'urgency': 5, 'complexity': 3}
        }
        
        scores = severity_scores.get(level, {'urgency': 3, 'complexity': 2})
        
        return ValueItem(
            id=item_id,
            title=f"Clippy {level}: {message.get('message', '')[:50]}...",
            description=message.get('message', ''),
            category=category,
            source="staticAnalysis",
            files_affected=[file_path],
            user_business_value=scores['urgency'] * 0.8,
            time_criticality=scores['urgency'] * 0.6,
            risk_reduction=scores['urgency'],
            opportunity_enablement=3.0,
            job_size=scores['complexity'],
            impact=scores['urgency'],
            confidence=8.0,
            ease=10 - scores['complexity'],
            debt_impact=scores['complexity'] * 1.5,
            debt_interest=scores['urgency'] * 0.5,
            hotspot_multiplier=1.2,
            estimated_hours=scores['complexity'] * 0.3
        )
    
    def _run_ruff_analysis(self) -> List[ValueItem]:
        """Run Python ruff analysis."""
        items = []
        
        try:
            result = subprocess.run([
                'ruff', 'check', '--output-format=json', 'python/'
            ], capture_output=True, text=True, cwd='.')
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues:
                    item = self._create_ruff_item(issue)
                    if item:
                        items.append(item)
                        
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _create_ruff_item(self, issue: Dict[str, Any]) -> Optional[ValueItem]:
        """Create a ValueItem from a ruff issue."""
        file_path = issue.get('filename', '')
        message = issue.get('message', '')
        code = issue.get('code', '')
        
        # Generate unique ID
        item_id = f"ruff-{hash(message + file_path + code) % 10000:04d}"
        
        # Categorize based on rule
        category = 'code_quality'
        severity = 3
        
        if code.startswith('S'):  # Security rules
            category = 'security'
            severity = 7
        elif code.startswith('PERF'):  # Performance rules
            category = 'performance' 
            severity = 5
        elif code.startswith('B'):  # Bugbear (potential bugs)
            category = 'bug_fix'
            severity = 6
        
        return ValueItem(
            id=item_id,
            title=f"Ruff {code}: {message[:50]}...",
            description=f"{code}: {message}",
            category=category,
            source="staticAnalysis",
            files_affected=[file_path],
            user_business_value=severity * 0.7,
            time_criticality=severity * 0.5,
            risk_reduction=severity * 0.9,
            opportunity_enablement=2.0,
            job_size=3.0,
            impact=severity,
            confidence=8.0,
            ease=7.0,
            debt_impact=severity,
            debt_interest=severity * 0.3,
            hotspot_multiplier=1.1,
            estimated_hours=0.5
        )
    
    def calculate_composite_scores(self):
        """Calculate composite scores for all discovered items."""
        weights = self.config['scoring']['weights']
        
        for item in self.discovered_items:
            # Calculate WSJF
            cost_of_delay = (
                item.user_business_value + 
                item.time_criticality + 
                item.risk_reduction + 
                item.opportunity_enablement
            )
            item.wsjf_score = cost_of_delay / max(item.job_size, 0.1)
            
            # Calculate ICE
            item.ice_score = item.impact * item.confidence * item.ease
            
            # Calculate Technical Debt Score
            item.technical_debt_score = (
                (item.debt_impact + item.debt_interest) * item.hotspot_multiplier
            )
            
            # Calculate Composite Score
            item.composite_score = (
                weights['wsjf'] * self._normalize_score(item.wsjf_score, 'wsjf') +
                weights['ice'] * self._normalize_score(item.ice_score, 'ice') +
                weights['technicalDebt'] * self._normalize_score(item.technical_debt_score, 'debt')
            )
            
            # Apply category boosts
            if item.category == 'security':
                item.composite_score *= self.config['scoring']['thresholds']['securityBoost']
            
            # Set priority based on composite score
            if item.composite_score >= 80:
                item.priority = 'high'
            elif item.composite_score >= 40:
                item.priority = 'medium'
            else:
                item.priority = 'low'
    
    def _normalize_score(self, score: float, score_type: str) -> float:
        """Normalize scores to 0-100 range."""
        if score_type == 'wsjf':
            return min(score * 10, 100)  # WSJF typically 0-10
        elif score_type == 'ice':
            return min(score / 10, 100)  # ICE typically 0-1000
        elif score_type == 'debt':
            return min(score * 5, 100)   # Debt score typically 0-20
        return score
    
    def run_discovery(self) -> List[ValueItem]:
        """Run the complete value discovery process."""
        print("🔍 Starting autonomous value discovery...")
        
        # Discover from configured sources
        sources = self.config.get('discovery', {}).get('sources', [])
        
        if 'gitHistory' in sources:
            print("  📚 Analyzing Git history...")
            self.discovered_items.extend(self.discover_from_git_history())
            
        if 'staticAnalysis' in sources:
            print("  🔬 Running static analysis...")
            self.discovered_items.extend(self.discover_from_static_analysis())
        
        # Calculate composite scores
        print("  🧮 Calculating value scores...")
        self.calculate_composite_scores()
        
        # Sort by composite score
        self.discovered_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        print(f"✅ Discovered {len(self.discovered_items)} value items")
        return self.discovered_items
    
    def export_items(self, output_path: str):
        """Export discovered items to JSON."""
        output = {
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_items': len(self.discovered_items),
                'discovery_sources': self.config.get('discovery', {}).get('sources', [])
            },
            'items': [asdict(item) for item in self.discovered_items]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"📄 Exported {len(self.discovered_items)} items to {output_path}")


def main():
    """Main entry point for value discovery."""
    parser = argparse.ArgumentParser(description="Autonomous Value Discovery Engine")
    parser.add_argument('--config', default='.terragon/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', default='.terragon/discovered-items.json',
                       help='Output path for discovered items')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize and run discovery
    engine = ValueDiscoveryEngine(args.config)
    items = engine.run_discovery()
    
    if args.verbose:
        print("\n📊 Top 5 Value Items:")
        for i, item in enumerate(items[:5], 1):
            print(f"  {i}. [{item.priority.upper()}] {item.title}")
            print(f"     Score: {item.composite_score:.1f} | "
                  f"Category: {item.category} | "
                  f"Est: {item.estimated_hours:.1f}h")
    
    # Export results
    engine.export_items(args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())