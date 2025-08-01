#!/usr/bin/env python3
"""
Autonomous Backlog Generator for liquid-audio-nets

Generates comprehensive backlogs with value scoring and prioritization
based on discovered items from the value discovery engine.
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class BacklogStats:
    """Statistics for the backlog."""
    total_items: int
    high_priority: int
    medium_priority: int
    low_priority: int
    avg_score: float
    categories: Dict[str, int]
    total_effort_hours: float


class BacklogGenerator:
    """Generates markdown backlogs from discovered value items."""
    
    def __init__(self, items_file: str):
        """Initialize with discovered items."""
        self.items_data = self._load_items(items_file)
        self.items = self.items_data.get('items', [])
        self.metadata = self.items_data.get('metadata', {})
        
    def _load_items(self, items_file: str) -> Dict[str, Any]:
        """Load discovered items from JSON file."""
        try:
            with open(items_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Items file {items_file} not found")
            return {'items': [], 'metadata': {}}
    
    def calculate_stats(self) -> BacklogStats:
        """Calculate backlog statistics."""
        if not self.items:
            return BacklogStats(0, 0, 0, 0, 0.0, {}, 0.0)
        
        priority_count = {'high': 0, 'medium': 0, 'low': 0}
        categories = {}
        total_score = 0.0
        total_hours = 0.0
        
        for item in self.items:
            priority = item.get('priority', 'medium')
            priority_count[priority] += 1
            
            category = item.get('category', 'other')
            categories[category] = categories.get(category, 0) + 1
            
            total_score += item.get('composite_score', 0.0)
            total_hours += item.get('estimated_hours', 0.0)
        
        avg_score = total_score / len(self.items) if self.items else 0.0
        
        return BacklogStats(
            total_items=len(self.items),
            high_priority=priority_count['high'],
            medium_priority=priority_count['medium'],
            low_priority=priority_count['low'],
            avg_score=avg_score,
            categories=categories,
            total_effort_hours=total_hours
        )
    
    def generate_markdown(self) -> str:
        """Generate comprehensive markdown backlog."""
        stats = self.calculate_stats()
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        md = []
        md.append("# 📊 Autonomous Value Backlog")
        md.append("")
        md.append(f"**Generated**: {timestamp}")
        md.append(f"**Total Items**: {stats.total_items}")
        md.append(f"**Total Estimated Effort**: {stats.total_effort_hours:.1f} hours")
        md.append("")
        
        # Discovery metadata
        if self.metadata:
            md.append("## 🔍 Discovery Information")
            md.append("")
            md.append(f"- **Discovery Sources**: {', '.join(self.metadata.get('discovery_sources', []))}")
            md.append(f"- **Items Discovered**: {self.metadata.get('total_items', 0)}")
            md.append(f"- **Last Discovery Run**: {self.metadata.get('generated_at', 'Unknown')}")
            md.append("")
        
        # Next best value item
        if self.items:
            next_item = self.items[0]
            md.append("## 🎯 Next Best Value Item")
            md.append("")
            md.append(f"**[{next_item['id']}] {next_item['title']}**")
            md.append("")
            md.append(f"- **Composite Score**: {next_item['composite_score']:.1f}")
            md.append(f"- **Priority**: {next_item['priority'].upper()}")
            md.append(f"- **Category**: {next_item['category']}")
            md.append(f"- **Estimated Effort**: {next_item['estimated_hours']:.1f} hours")
            md.append(f"- **Source**: {next_item['source']}")
            md.append("")
            md.append(f"**Description**: {next_item['description']}")
            md.append("")
            if next_item.get('files_affected'):
                md.append(f"**Files Affected**: {', '.join(next_item['files_affected'])}")
                md.append("")
        
        # Summary statistics
        md.append("## 📈 Backlog Statistics")
        md.append("")
        md.append("### Priority Distribution")
        md.append("")
        md.append(f"- 🔴 **High Priority**: {stats.high_priority} items")
        md.append(f"- 🟡 **Medium Priority**: {stats.medium_priority} items") 
        md.append(f"- 🟢 **Low Priority**: {stats.low_priority} items")
        md.append("")
        
        md.append("### Category Breakdown")
        md.append("")
        for category, count in sorted(stats.categories.items(), key=lambda x: x[1], reverse=True):
            emoji = self._get_category_emoji(category)
            md.append(f"- {emoji} **{category.replace('_', ' ').title()}**: {count} items")
        md.append("")
        
        # Top items table
        md.append("## 📋 Top Priority Items")
        md.append("")
        md.append("| Rank | ID | Title | Score | Priority | Category | Hours |")
        md.append("|------|-----|--------|--------|----------|----------|-------|")
        
        for i, item in enumerate(self.items[:20], 1):  # Top 20 items
            title = item['title'][:50] + "..." if len(item['title']) > 50 else item['title']
            priority_emoji = self._get_priority_emoji(item['priority'])
            
            md.append(f"| {i} | `{item['id']}` | {title} | {item['composite_score']:.1f} | "
                     f"{priority_emoji} {item['priority']} | {item['category']} | {item['estimated_hours']:.1f} |")
        
        md.append("")
        
        # Detailed item breakdown by priority
        for priority in ['high', 'medium', 'low']:
            priority_items = [item for item in self.items if item.get('priority') == priority]
            if not priority_items:
                continue
                
            priority_emoji = self._get_priority_emoji(priority)
            md.append(f"## {priority_emoji} {priority.title()} Priority Items")
            md.append("")
            
            for item in priority_items:
                md.append(f"### [{item['id']}] {item['title']}")
                md.append("")
                md.append(f"**Score**: {item['composite_score']:.1f} | "
                         f"**Category**: {item['category']} | "
                         f"**Effort**: {item['estimated_hours']:.1f}h | "
                         f"**Source**: {item['source']}")
                md.append("")
                md.append(f"{item['description']}")
                md.append("")
                
                # Show scoring breakdown for high priority items
                if priority == 'high':
                    md.append("**Scoring Breakdown:**")
                    md.append(f"- WSJF Score: {item.get('wsjf_score', 0):.1f}")
                    md.append(f"- ICE Score: {item.get('ice_score', 0):.1f}")
                    md.append(f"- Technical Debt Score: {item.get('technical_debt_score', 0):.1f}")
                    md.append("")
                
                if item.get('files_affected'):
                    md.append(f"**Files**: {', '.join(item['files_affected'])}")
                    md.append("")
                
                md.append("---")
                md.append("")
        
        # Value metrics summary
        md.append("## 📊 Value Metrics")
        md.append("")
        md.append(f"- **Average Composite Score**: {stats.avg_score:.1f}")
        md.append(f"- **Total Value Potential**: {sum(item.get('composite_score', 0) for item in self.items):.1f}")
        md.append(f"- **High-Value Items (>80 score)**: {len([i for i in self.items if i.get('composite_score', 0) > 80])}")
        md.append(f"- **Quick Wins (<2h effort)**: {len([i for i in self.items if i.get('estimated_hours', 0) < 2])}")
        md.append("")
        
        # Implementation recommendations
        md.append("## 🚀 Implementation Recommendations")
        md.append("")
        
        high_items = [i for i in self.items if i.get('priority') == 'high']
        quick_wins = [i for i in self.items if i.get('estimated_hours', 0) < 2 and i.get('composite_score', 0) > 40]
        security_items = [i for i in self.items if i.get('category') == 'security']
        
        if high_items:
            md.append("### Immediate Action Items")
            md.append("")
            md.append("Start with these high-priority items for maximum value:")
            for item in high_items[:3]:
                md.append(f"1. **{item['title']}** (Score: {item['composite_score']:.1f}, {item['estimated_hours']:.1f}h)")
            md.append("")
        
        if quick_wins:
            md.append("### Quick Wins")
            md.append("")
            md.append("Low-effort, high-impact items for immediate delivery:")
            for item in quick_wins[:3]:
                md.append(f"- **{item['title']}** (Score: {item['composite_score']:.1f}, {item['estimated_hours']:.1f}h)")
            md.append("")
        
        if security_items:
            md.append("### Security Focus")
            md.append("")
            md.append(f"{len(security_items)} security-related items require attention:")
            for item in security_items[:3]:
                md.append(f"- **{item['title']}** (Score: {item['composite_score']:.1f})")
            md.append("")
        
        # Automation notes
        md.append("## 🤖 Autonomous Execution")
        md.append("")
        md.append("This backlog is continuously updated through autonomous value discovery. ")
        md.append("Items are automatically:")
        md.append("")
        md.append("- Discovered from multiple sources (Git history, static analysis, monitoring)")
        md.append("- Scored using WSJF, ICE, and technical debt metrics")
        md.append("- Prioritized based on composite value scores")
        md.append("- Updated as the repository evolves")
        md.append("")
        md.append("The next highest-value item will be automatically selected for implementation ")
        md.append("when the current work is completed.")
        md.append("")
        
        # Footer
        md.append("---")
        md.append("")
        md.append("*Generated by Terragon Autonomous SDLC System*")
        md.append("")
        
        return "\n".join(md)
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category."""
        emojis = {
            'security': '🔒',
            'performance': '⚡',
            'technical_debt': '🔧',
            'bug_fix': '🐛',
            'feature': '✨',
            'testing': '🧪',
            'documentation': '📚',
            'code_quality': '🎯',
            'infrastructure': '🏗️'
        }
        return emojis.get(category, '📝')
    
    def _get_priority_emoji(self, priority: str) -> str:
        """Get emoji for priority."""
        emojis = {
            'high': '🔴',
            'medium': '🟡', 
            'low': '🟢'
        }
        return emojis.get(priority, '⚪')
    
    def generate_json_summary(self) -> Dict[str, Any]:
        """Generate JSON summary for API consumption."""
        stats = self.calculate_stats()
        
        return {
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_items': stats.total_items,
                'total_effort_hours': stats.total_effort_hours,
                'average_score': stats.avg_score
            },
            'statistics': {
                'priority_distribution': {
                    'high': stats.high_priority,
                    'medium': stats.medium_priority,
                    'low': stats.low_priority
                },
                'category_distribution': stats.categories
            },
            'next_best_item': self.items[0] if self.items else None,
            'top_items': self.items[:10] if self.items else [],
            'quick_wins': [
                item for item in self.items 
                if item.get('estimated_hours', 0) < 2 and item.get('composite_score', 0) > 40
            ][:5],
            'security_items': [
                item for item in self.items 
                if item.get('category') == 'security'
            ][:5]
        }


def main():
    """Main entry point for backlog generation."""
    parser = argparse.ArgumentParser(description="Autonomous Backlog Generator")
    parser.add_argument('--input', required=True,
                       help='Input JSON file with discovered items')
    parser.add_argument('--output', default='BACKLOG.md',
                       help='Output markdown file')
    parser.add_argument('--format', choices=['markdown', 'json'], default='markdown',
                       help='Output format')
    parser.add_argument('--json-output', help='JSON output file (when format=json)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = BacklogGenerator(args.input)
    
    if args.format == 'markdown':
        # Generate markdown backlog
        markdown = generator.generate_markdown()
        
        # Write to file
        with open(args.output, 'w') as f:
            f.write(markdown)
        
        print(f"📄 Generated markdown backlog: {args.output}")
        
        # Also generate JSON summary if requested
        if args.json_output:
            summary = generator.generate_json_summary()
            with open(args.json_output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📊 Generated JSON summary: {args.json_output}")
    
    elif args.format == 'json':
        # Generate JSON summary
        summary = generator.generate_json_summary()
        output_file = args.json_output or args.output
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Generated JSON backlog: {output_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())