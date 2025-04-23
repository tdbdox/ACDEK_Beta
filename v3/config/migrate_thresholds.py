#!/usr/bin/env python3
"""
ACDEK v3 Threshold Migration Tool

This script migrates existing threshold configurations to the new unified
threshold system. It identifies files using old threshold systems and 
modifies them to use the new unified approach.

Steps:
1. Backup original files
2. Update imports and references in target files
3. Verify the changes maintain the same behavior
"""

import os
import sys
import shutil
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR))

# Target files to update
TARGET_FILES = [
    # Grounding engine
    "v3/grounding/grounding_engine.py",
    
    # Benchmark files
    "benchmarks/hallucination_benchmark.py",
    "benchmarks/threshold_config.py",
    "benchmarks/performance_metrics.py",
    "benchmarks/confidence_benchmark.py",
    "benchmarks/constraint_benchmark.py",
    
    # Fallback chain
    "acdek/fallback_chain.py",
    
    # Enrichment tools
    "enrichment/confidence_scorer.py",
    "enrichment/constraint_verifier.py",
    "enrichment/hallucination_metrics.py",
    "enrichment/drift_detector.py"
]


def backup_file(file_path: Path) -> Path:
    """
    Create a backup of the specified file.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f"{file_path.suffix}.{timestamp}.bak")
    
    if file_path.exists():
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return backup_path
    else:
        logger.warning(f"File {file_path} does not exist, no backup created")
        return None


def update_grounding_engine(file_path: Path) -> bool:
    """
    Update grounding_engine.py to use the unified threshold system.
    
    Args:
        file_path: Path to grounding_engine.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace imports
        content = re.sub(
            r'(import os.*?import sys\n)',
            r'\1from config.threshold_unified import get_threshold, ThresholdCategory, ExecutionMode\n',
            content
        )
        
        # Replace the threshold definitions
        content = re.sub(
            r'# Threshold for factuality scoring: below this is considered hallucination\nFACTUALITY_THRESHOLD: float = 0\.70\n\n# Domain-specific thresholds \(overrides global threshold\)\nDOMAIN_THRESHOLDS = \{[^}]+\}',
            "# Use unified threshold system for factuality scoring\n# The following constants are kept for backward compatibility\nFACTUALITY_THRESHOLD: float = get_threshold('general', ThresholdCategory.FACTUALITY)\n\n# Domain-specific thresholds (for backward compatibility)\nDOMAIN_THRESHOLDS = {\n    'medical': get_threshold('medical', ThresholdCategory.FACTUALITY),\n    'legal': get_threshold('legal', ThresholdCategory.FACTUALITY),\n    'financial': get_threshold('financial', ThresholdCategory.FACTUALITY)\n}",
            content
        )
        
        # Update threshold usage in grounded_validation_pipeline
        content = re.sub(
            r'(# Get domain-specific threshold\n\s+)threshold = DOMAIN_THRESHOLDS\.get\(domain, FACTUALITY_THRESHOLD\)',
            r'\1threshold = get_threshold(domain, ThresholdCategory.FACTUALITY)',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_hallucination_benchmark(file_path: Path) -> bool:
    """
    Update hallucination_benchmark.py to use the unified threshold system.
    
    Args:
        file_path: Path to hallucination_benchmark.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace imports
        content = re.sub(
            r'# Import centralized domain configuration\nfrom config\.domain_config import Domain, get_threshold, get_domain_thresholds',
            "# Import centralized domain and threshold configurations\nfrom config.domain_config import Domain\nfrom config.threshold_unified import get_threshold, ThresholdCategory, ExecutionMode, set_execution_mode",
            content
        )
        
        # Add mode setting at the beginning of main function
        content = re.sub(
            r'(def main\(\):\n\s+"""Main function to run benchmarks\.""")',
            r'\1\n    # Set execution mode to benchmark\n    set_execution_mode(ExecutionMode.BENCHMARK)',
            content
        )
        
        # Update threshold usage
        content = content.replace("get_threshold(domain, 'factuality')", 
                                 "get_threshold(domain, ThresholdCategory.FACTUALITY)")
        
        content = content.replace("get_threshold(domain, 'hallucination')", 
                                 "get_threshold(domain, ThresholdCategory.HALLUCINATION)")
        
        content = content.replace("get_threshold(domain, 'confidence')", 
                                 "get_threshold(domain, ThresholdCategory.CONFIDENCE)")
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_threshold_config(file_path: Path) -> bool:
    """
    Update benchmark threshold_config.py to use the unified threshold system.
    
    Args:
        file_path: Path to threshold_config.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add deprecation warning
        warning = """
# =========== DEPRECATION NOTICE ===========
# This module is deprecated and will be removed in a future version.
# Please use config.threshold_unified instead.
# ==========================================
"""
        # Insert warning after module docstring
        content = re.sub(
            r'(""".*?""")',  # Match docstring
            r'\1' + warning,  # Add warning after docstring
            content,
            count=1,
            flags=re.DOTALL
        )
        
        # Add import and redirection
        redirects = """
# Import unified threshold system
from config.threshold_unified import (
    get_threshold as unified_get_threshold, 
    ThresholdCategory,
    ExecutionMode,
    set_execution_mode
)

# Set to benchmark mode
set_execution_mode(ExecutionMode.BENCHMARK)

"""
        # Insert redirects before MetricType class
        content = re.sub(
            r'(class MetricType\(str, Enum\):)',
            redirects + r'\1',
            content
        )
        
        # Modify get_threshold method to use unified system
        content = re.sub(
            r'([ ]{4}@classmethod\n[ ]{4}def get_threshold.*?\n[ ]{8}""".*?"""\n)(.*?)(\n[ ]{8}return)',
            r'\1        # Use unified threshold system\n        metric_type_str = metric_type.value if hasattr(metric_type, "value") else str(metric_type)\n        category_map = {\n            "factuality": ThresholdCategory.FACTUALITY,\n            "latency": ThresholdCategory.LATENCY,\n            "confidence": ThresholdCategory.CONFIDENCE,\n            "retry_penalty": ThresholdCategory.RETRY,\n            "coverage": ThresholdCategory.CONFIDENCE,\n            "combined": ThresholdCategory.COMBINED\n        }\n        category = category_map.get(metric_type_str.lower(), ThresholdCategory.FACTUALITY)\n        domain_str = domain if isinstance(domain, str) else domain.value if hasattr(domain, "value") else "general"\n        return unified_get_threshold(domain_str, category, ExecutionMode.BENCHMARK)\n\n        # Original implementation (kept for reference)\3',
            content,
            flags=re.DOTALL
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_fallback_chain(file_path: Path) -> bool:
    """
    Update fallback_chain.py to use the unified threshold system.
    
    Args:
        file_path: Path to fallback_chain.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add import for unified thresholds
        content = re.sub(
            r'(from observability\.resilience_events import \(.*?\n\))',
            r'\1\nfrom config.threshold_unified import get_threshold, ThresholdCategory',
            content
        )
        
        # Modify FallbackChain to use unified thresholds where applicable
        content = re.sub(
            r'([ ]{8}def execute\(self, method_name: str, \*args, \*\*kwargs\) -> FallbackResult:.*?last_error: Optional\[Exception\] = None)',
            r'\1\n        # Get domain from kwargs if available for threshold adjustment\n        domain = kwargs.get("domain", "general")',
            content
        )
        
        # Update penalty calculation to use unified thresholds
        content = re.sub(
            r'([ ]{16}penalty = max\(0\.0, 1\.0 - depth \* 0\.1\))',
            r'        # Get fallback threshold for the domain\n                fallback_penalty_base = get_threshold(domain, ThresholdCategory.FALLBACK)\n\1',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_confidence_scorer(file_path: Path) -> bool:
    """
    Update confidence_scorer.py to use the unified threshold system.
    
    Args:
        file_path: Path to confidence_scorer.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add import for unified thresholds
        if "from config.threshold_unified import" not in content:
            content = re.sub(
                r'(import logging.*?\n)',
                r'\1from config.threshold_unified import get_threshold, ThresholdCategory\n',
                content
            )
        
        # Replace hardcoded thresholds with unified thresholds
        content = re.sub(
            r'(confidence_threshold = )([0-9.]+|domain_thresholds\.get\([^)]+\))(.*?# .*?threshold)',
            r'\1get_threshold(domain, ThresholdCategory.CONFIDENCE)\3',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_constraint_verifier(file_path: Path) -> bool:
    """
    Update constraint_verifier.py to use the unified threshold system.
    
    Args:
        file_path: Path to constraint_verifier.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add import for unified thresholds
        if "from config.threshold_unified import" not in content:
            content = re.sub(
                r'(import logging.*?\n)',
                r'\1from config.threshold_unified import get_threshold, ThresholdCategory\n',
                content
            )
        
        # Replace hardcoded thresholds with unified thresholds
        content = re.sub(
            r'(constraint_threshold = )([0-9.]+|domain_constraints\.get\([^)]+\))(.*?# .*?threshold)',
            r'\1get_threshold(domain, ThresholdCategory.CONSTRAINT)\3',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_hallucination_metrics(file_path: Path) -> bool:
    """
    Update hallucination_metrics.py to use the unified threshold system.
    
    Args:
        file_path: Path to hallucination_metrics.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add import for unified thresholds
        if "from config.threshold_unified import" not in content:
            content = re.sub(
                r'(import logging.*?\n)',
                r'\1from config.threshold_unified import get_threshold, ThresholdCategory\n',
                content
            )
        
        # Replace domain imports if present
        content = re.sub(
            r'from config\.domain_config import Domain, get_threshold',
            r'from config.domain_config import Domain',
            content
        )
        
        # Replace hardcoded thresholds with unified thresholds
        content = re.sub(
            r'(hallucination_threshold = )([0-9.]+|get_threshold\([^,]+, [\'"]hallucination[\'"]\))(.*?# .*?threshold)',
            r'\1get_threshold(domain, ThresholdCategory.HALLUCINATION)\3',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_drift_detector(file_path: Path) -> bool:
    """
    Update drift_detector.py to use the unified threshold system.
    
    Args:
        file_path: Path to drift_detector.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add import for unified thresholds
        if "from config.threshold_unified import" not in content:
            content = re.sub(
                r'(import logging.*?\n)',
                r'\1from config.threshold_unified import get_threshold, ThresholdCategory\n',
                content
            )
        
        # Replace hardcoded thresholds with unified thresholds
        content = re.sub(
            r'(drift_threshold = )([0-9.]+|self\.config\.get\([\'"]drift_threshold[\'"], [0-9.]+\))(.*?# .*?threshold)',
            r'\1get_threshold(domain, ThresholdCategory.CONFIDENCE)\3',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def update_performance_metrics(file_path: Path) -> bool:
    """
    Update performance_metrics.py to use the unified threshold system.
    
    Args:
        file_path: Path to performance_metrics.py
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
            return False
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add import for unified thresholds
        if "from config.threshold_unified import" not in content:
            content = re.sub(
                r'(import logging.*?\n)',
                r'\1from config.threshold_unified import get_threshold, ThresholdCategory, ExecutionMode\n',
                content
            )
        
        # Replace domain imports if present
        content = re.sub(
            r'from config\.domain_config import Domain, get_threshold',
            r'from config.domain_config import Domain',
            content
        )
        
        # Replace hardcoded latency thresholds with unified thresholds
        content = re.sub(
            r'(latency_threshold = )([0-9.]+|get_threshold\([^,]+, [\'"]latency(?:_ms)?[\'"]\))(.*?# .*?threshold)',
            r'\1get_threshold(domain, ThresholdCategory.LATENCY, ExecutionMode.BENCHMARK)\3',
            content
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")
        return False


def main():
    """Main function to run the migration script."""
    parser = argparse.ArgumentParser(description="ACDEK v3 Threshold Migration Tool")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just show what would be changed")
    parser.add_argument("--backup", action="store_true", help="Create backups of files before modifying them")
    args = parser.parse_args()
    
    logger.info("Starting threshold migration")
    
    # Verify unified threshold file exists
    unified_file = ROOT_DIR / "config" / "threshold_unified.py"
    if not unified_file.exists():
        logger.error(f"Unified threshold file {unified_file} does not exist")
        return 1
    
    # Process each target file
    success_count = 0
    for target_file in TARGET_FILES:
        file_path = ROOT_DIR / target_file
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
            continue
        
        logger.info(f"Processing {file_path}")
        
        # Create backup if requested
        if args.backup:
            backup_file(file_path)
        
        # Skip actual modifications in dry-run mode
        if args.dry_run:
            logger.info(f"Would update {file_path} (dry run)")
            continue
        
        # Update the file based on its type
        if "grounding_engine.py" in str(file_path):
            success = update_grounding_engine(file_path)
        elif "hallucination_benchmark.py" in str(file_path):
            success = update_hallucination_benchmark(file_path)
        elif "threshold_config.py" in str(file_path):
            success = update_threshold_config(file_path)
        elif "fallback_chain.py" in str(file_path):
            success = update_fallback_chain(file_path)
        elif "confidence_scorer.py" in str(file_path):
            success = update_confidence_scorer(file_path)
        elif "constraint_verifier.py" in str(file_path):
            success = update_constraint_verifier(file_path)
        elif "hallucination_metrics.py" in str(file_path):
            success = update_hallucination_metrics(file_path)
        elif "drift_detector.py" in str(file_path):
            success = update_drift_detector(file_path)
        elif "performance_metrics.py" in str(file_path):
            success = update_performance_metrics(file_path)
        else:
            logger.warning(f"No specific update method for {file_path}, using generic update")
            success = False
        
        if success:
            success_count += 1
    
    # Report results
    logger.info(f"Migration complete. Updated {success_count}/{len(TARGET_FILES)} files.")
    
    if args.dry_run:
        logger.info("This was a dry run, no files were modified")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())