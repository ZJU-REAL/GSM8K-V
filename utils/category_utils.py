"""
Data category management utility for GSM8K-V evaluation.

This script provides utilities to:
1. List available data categories and subcategories
2. Analyze data distribution across categories
3. Generate category-specific evaluation configurations
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

from config.data_category_config import (
    DataCategory, DataSubCategory, DataCategoryConfig, 
    create_category_config_from_names
)
from utils.data_loader import load_category_filtered_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_categories():
    """List all available data categories and subcategories."""
    print("=== Available Data Categories ===")
    
    config = DataCategoryConfig()
    
    for category in DataCategory:
        subcategories = config.CATEGORY_SUBCATEGORY_MAP[category]
        print(f"\n{category.value.upper()}:")
        
        for subcategory in subcategories:
            file_path = config.SUBCATEGORY_FILE_MAP[subcategory]
            file_exists = os.path.exists(os.path.join(config.metadata_dir, file_path))
            status = "✓" if file_exists else "✗"
            print(f"  {status} {subcategory.value} -> {file_path}")


def analyze_category_data(metadata_dir: str = "data/metadata"):
    """Analyze data distribution across all categories."""
    print("=== Data Category Analysis ===")
    
    config = DataCategoryConfig(metadata_dir=metadata_dir)
    enabled_files = config.get_enabled_data_files()
    
    total_samples = 0
    category_counts = defaultdict(int)
    subcategory_counts = defaultdict(int)
    
    for subcategory, file_path in enabled_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sample_count = len(data)
            total_samples += sample_count
            
            parent_category = config._get_parent_category(subcategory)
            category_counts[parent_category.value] += sample_count
            subcategory_counts[subcategory.value] = sample_count
            
            print(f"{subcategory.value}: {sample_count:,} samples")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"\nTotal samples across all categories: {total_samples:,}")
    
    print("\n=== Category Distribution ===")
    for category, count in sorted(category_counts.items()):
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{category}: {count:,} samples ({percentage:.1f}%)")
    
    print("\n=== Subcategory Distribution ===")
    for subcategory, count in sorted(subcategory_counts.items()):
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{subcategory}: {count:,} samples ({percentage:.1f}%)")


def sample_category_data(
    categories: Optional[List[str]] = None,
    subcategories: Optional[List[str]] = None,
    num_samples: int = 5,
    metadata_dir: str = "data/metadata"
):
    """Sample data from specified categories for inspection."""
    print(f"=== Sampling {num_samples} examples from specified categories ===")
    
    try:
        category_config = create_category_config_from_names(
            category_names=categories,
            subcategory_names=subcategories,
            metadata_dir=metadata_dir
        )
        
        data = load_category_filtered_data(category_config, num_samples)
        
        if not data:
            print("No data found for specified categories.")
            return
        
        # Group samples by category
        category_samples = defaultdict(list)
        for sample in data:
            category = sample.get('data_category', 'unknown')
            category_samples[category].append(sample)
        
        for category, samples in category_samples.items():
            print(f"\n--- {category.upper()} ---")
            for i, sample in enumerate(samples[:num_samples]):
                print(f"\nSample {i+1}:")
                print(f"  ID: {sample.get('question_id', 'N/A')}")
                print(f"  Subcategory: {sample.get('data_subcategory', 'N/A')}")
                print(f"  Question: {sample.get('original_question', 'N/A')[:100]}...")
                print(f"  Answer: {sample.get('math_ground_truth', 'N/A')}")
                print(f"  Images: {len(sample.get('pic_ids', []))} files")
        
    except ValueError as e:
        print(f"Error: {e}")


def generate_config(
    categories: Optional[List[str]] = None,
    subcategories: Optional[List[str]] = None,
    output_file: str = "generated_config.json",
    base_config: str = "config.json"
):
    """Generate a configuration file for category-specific evaluation."""
    print(f"=== Generating configuration for category evaluation ===")
    
    try:
        # Load base configuration
        with open(base_config, 'r') as f:
            config = json.load(f)
        
        # Add category configuration
        if categories:
            config["data_categories"] = categories
            print(f"Enabled categories: {categories}")
        
        if subcategories:
            config["data_subcategories"] = subcategories
            print(f"Enabled subcategories: {subcategories}")
        
        # Save new configuration
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {output_file}")
        
        # Display summary
        try:
            category_config = create_category_config_from_names(
                category_names=categories,
                subcategory_names=subcategories,
                metadata_dir=config.get("metadata_dir", "data/metadata")
            )
            
            print(f"Configuration summary:")
            print(f"  Enabled categories: {[cat.value for cat in category_config.enabled_categories]}")
            print(f"  Enabled subcategories: {[subcat.value for subcat in category_config.enabled_subcategories]}")
            print(f"  Data files: {len(category_config.get_enabled_data_files())}")
            
        except Exception as e:
            print(f"Warning: Could not validate configuration: {e}")
        
    except Exception as e:
        print(f"Error generating configuration: {e}")


def main():
    """Main function for the data category utility."""
    parser = argparse.ArgumentParser(
        description="Data category management utility for GSM8K-V evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python category_utils.py list                          # List all categories
  python category_utils.py analyze                       # Analyze data distribution
  python category_utils.py sample --categories measurement --num-samples 3
  python category_utils.py generate --subcategories distance price --output my_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available categories and subcategories')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data distribution across categories')
    analyze_parser.add_argument('--metadata-dir', default='data/metadata', help='Metadata directory path')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Sample data from specified categories')
    sample_parser.add_argument('--categories', nargs='+', help='Categories to sample from')
    sample_parser.add_argument('--subcategories', nargs='+', help='Subcategories to sample from')
    sample_parser.add_argument('--num-samples', type=int, default=5, help='Number of samples per category')
    sample_parser.add_argument('--metadata-dir', default='data/metadata', help='Metadata directory path')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate configuration for category evaluation')
    generate_parser.add_argument('--categories', nargs='+', help='Categories to enable')
    generate_parser.add_argument('--subcategories', nargs='+', help='Subcategories to enable')
    generate_parser.add_argument('--output', default='generated_config.json', help='Output configuration file')
    generate_parser.add_argument('--base-config', default='config.json', help='Base configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_categories()
    elif args.command == 'analyze':
        analyze_category_data(args.metadata_dir)
    elif args.command == 'sample':
        sample_category_data(
            categories=args.categories,
            subcategories=args.subcategories,
            num_samples=args.num_samples,
            metadata_dir=args.metadata_dir
        )
    elif args.command == 'generate':
        generate_config(
            categories=args.categories,
            subcategories=args.subcategories,
            output_file=args.output,
            base_config=args.base_config
        )


if __name__ == "__main__":
    main()
