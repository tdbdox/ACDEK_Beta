#!/usr/bin/env python3
"""
Test script for the unified threshold system.

This script verifies that the unified threshold system works correctly
by comparing threshold values across different domains, categories, and modes.
"""

import sys
import os
from pathlib import Path
import logging

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the components we need to test
from config.domain_config import Domain
from config.threshold_unified import (
    get_threshold, 
    ThresholdCategory, 
    ExecutionMode, 
    set_execution_mode,
    set_threshold_override,
    clear_threshold_overrides,
    export_configuration,
    import_configuration
)

def test_domain_thresholds():
    """Test thresholds across different domains."""
    print("\n=== Testing Domain Thresholds ===")
    
    # Set to production mode for consistent testing
    set_execution_mode(ExecutionMode.PRODUCTION)
    
    # Test factuality thresholds for each domain
    domains = [
        Domain.MEDICAL,
        Domain.LEGAL,
        Domain.FINANCIAL,
        Domain.SCIENTIFIC,
        Domain.TECHNICAL,
        Domain.HISTORICAL,
        Domain.ACADEMIC,
        Domain.GENERAL
    ]
    
    for domain in domains:
        threshold = get_threshold(domain, ThresholdCategory.FACTUALITY)
        print(f"{domain.value} factuality threshold: {threshold}")
    
    # Verify that some domains have stricter thresholds than others
    med_threshold = get_threshold(Domain.MEDICAL, ThresholdCategory.FACTUALITY)
    gen_threshold = get_threshold(Domain.GENERAL, ThresholdCategory.FACTUALITY)
    
    assert med_threshold > gen_threshold, "Medical domain should have stricter threshold than General"
    print("✓ Medical domain correctly has stricter threshold than General domain")

def test_execution_modes():
    """Test thresholds across different execution modes."""
    print("\n=== Testing Execution Modes ===")
    
    domain = Domain.MEDICAL
    category = ThresholdCategory.FACTUALITY
    
    # Test each execution mode
    modes = [
        ExecutionMode.PRODUCTION,
        ExecutionMode.BENCHMARK,
        ExecutionMode.DEVELOPMENT,
        ExecutionMode.TESTING
    ]
    
    for mode in modes:
        set_execution_mode(mode)
        threshold = get_threshold(domain, category)
        print(f"{mode.value} mode {domain.value} {category.value} threshold: {threshold}")
    
    # Verify that production has stricter thresholds than testing
    set_execution_mode(ExecutionMode.PRODUCTION)
    prod_threshold = get_threshold(domain, category)
    
    set_execution_mode(ExecutionMode.TESTING)
    test_threshold = get_threshold(domain, category)
    
    assert prod_threshold > test_threshold, "Production mode should have stricter threshold than Testing mode"
    print("✓ Production mode correctly has stricter threshold than Testing mode")

def test_threshold_categories():
    """Test different threshold categories."""
    print("\n=== Testing Threshold Categories ===")
    
    # Set to production mode
    set_execution_mode(ExecutionMode.PRODUCTION)
    domain = Domain.MEDICAL
    
    # Test each category
    categories = [
        ThresholdCategory.FACTUALITY,
        ThresholdCategory.HALLUCINATION,
        ThresholdCategory.CONFIDENCE,
        ThresholdCategory.CONSTRAINT,
        ThresholdCategory.LATENCY,
        ThresholdCategory.RETRY,
        ThresholdCategory.FALLBACK,
        ThresholdCategory.COMBINED
    ]
    
    for category in categories:
        threshold = get_threshold(domain, category)
        print(f"{domain.value} {category.value} threshold: {threshold}")
    
    # Verify that constraint thresholds are typically higher than confidence thresholds
    constraint_threshold = get_threshold(domain, ThresholdCategory.CONSTRAINT)
    confidence_threshold = get_threshold(domain, ThresholdCategory.CONFIDENCE)
    
    assert constraint_threshold > confidence_threshold, "Constraint threshold should be higher than confidence threshold"
    print("✓ Constraint threshold correctly higher than confidence threshold")

def test_overrides():
    """Test threshold overrides."""
    print("\n=== Testing Threshold Overrides ===")
    
    # Set to production mode
    set_execution_mode(ExecutionMode.PRODUCTION)
    domain = Domain.MEDICAL
    category = ThresholdCategory.FACTUALITY
    
    # Get original threshold
    original_threshold = get_threshold(domain, category)
    print(f"Original {domain.value} {category.value} threshold: {original_threshold}")
    
    # Set override
    override_value = 0.95
    set_threshold_override(domain, category, override_value)
    
    # Verify override is applied
    new_threshold = get_threshold(domain, category)
    print(f"Overridden {domain.value} {category.value} threshold: {new_threshold}")
    
    assert new_threshold == override_value, "Override value should be applied"
    print("✓ Override correctly applied")
    
    # Clear overrides
    clear_threshold_overrides()
    
    # Verify original threshold is restored
    restored_threshold = get_threshold(domain, category)
    print(f"Restored {domain.value} {category.value} threshold: {restored_threshold}")
    
    assert abs(restored_threshold - original_threshold) < 0.001, "Original threshold should be restored after clearing overrides"
    print("✓ Original threshold correctly restored after clearing overrides")

def test_configuration_export_import():
    """Test configuration export and import."""
    print("\n=== Testing Configuration Export/Import ===")
    
    # Set to production mode
    set_execution_mode(ExecutionMode.PRODUCTION)
    
    # Set some overrides
    set_threshold_override(Domain.MEDICAL, ThresholdCategory.FACTUALITY, 0.95)
    set_threshold_override(Domain.LEGAL, ThresholdCategory.HALLUCINATION, 0.92)
    
    # Export configuration
    temp_file = Path("./temp_thresholds.json")
    export_configuration(temp_file)
    print(f"Configuration exported to {temp_file}")
    
    # Clear overrides
    clear_threshold_overrides()
    
    # Import configuration
    import_configuration(temp_file)
    print(f"Configuration imported from {temp_file}")
    
    # Verify overrides were restored
    med_threshold = get_threshold(Domain.MEDICAL, ThresholdCategory.FACTUALITY)
    legal_threshold = get_threshold(Domain.LEGAL, ThresholdCategory.HALLUCINATION)
    
    assert abs(med_threshold - 0.95) < 0.001, "Medical factuality override should be restored"
    assert abs(legal_threshold - 0.92) < 0.001, "Legal hallucination override should be restored"
    print("✓ Overrides correctly restored from imported configuration")
    
    # Clean up
    os.remove(temp_file)
    clear_threshold_overrides()
    print(f"Temporary configuration file removed")

def main():
    """Run all tests."""
    print("Testing ACDEK v3 Unified Threshold System")
    
    try:
        test_domain_thresholds()
        test_execution_modes()
        test_threshold_categories()
        test_overrides()
        test_configuration_export_import()
        
        print("\n✓ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())