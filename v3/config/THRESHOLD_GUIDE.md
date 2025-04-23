# ACDEK v3 Unified Threshold System

## Overview

The ACDEK v3 Unified Threshold System provides a centralized approach to managing thresholds across all components of the platform. This system ensures consistency between benchmark modules and production code, eliminating the threshold configuration inconsistencies identified in the validation report.

## Key Features

- **Single Source of Truth**: All threshold values are defined in one place, eliminating inconsistencies
- **Domain-Specific Configuration**: Different thresholds for medical, legal, financial, and other domains
- **Mode-Specific Adjustments**: Different threshold values for production, benchmark, development, and testing environments
- **Runtime Overrides**: Dynamic threshold adjustments for specific scenarios
- **Migration Path**: Compatibility layer for existing code that uses the old threshold systems
- **Import/Export Capability**: Configuration management for saving and loading threshold settings

## Using the Unified Threshold System

### Basic Usage

```python
from config.threshold_unified import get_threshold, ThresholdCategory, ExecutionMode

# Get the factuality threshold for the medical domain in production mode
threshold = get_threshold(domain="medical", category=ThresholdCategory.FACTUALITY)

# Get the hallucination threshold for the legal domain in benchmark mode
benchmark_threshold = get_threshold(
    domain="legal", 
    category=ThresholdCategory.HALLUCINATION,
    mode=ExecutionMode.BENCHMARK
)

# Check if a score exceeds the threshold
if factuality_score >= threshold:
    # Handle valid response
else:
    # Handle potential hallucination
```

### Setting Execution Mode

The system automatically detects the execution environment, but you can explicitly set the mode:

```python
from config.threshold_unified import set_execution_mode, ExecutionMode

# Set to benchmark mode
set_execution_mode(ExecutionMode.BENCHMARK)

# Set to production mode
set_execution_mode(ExecutionMode.PRODUCTION)

# Set to development mode
set_execution_mode(ExecutionMode.DEVELOPMENT)

# Set to testing mode
set_execution_mode(ExecutionMode.TESTING)
```

### Runtime Overrides

For special scenarios, you can override threshold values at runtime:

```python
from config.threshold_unified import set_threshold_override, ThresholdCategory

# Set a more strict factuality threshold for a specific high-risk domain
set_threshold_override(
    domain="medical",
    category=ThresholdCategory.FACTUALITY,
    value=0.90  # More strict than default
)

# Clear all overrides
from config.threshold_unified import clear_threshold_overrides
clear_threshold_overrides()
```

### Threshold Categories

The system supports the following threshold categories:

- `ThresholdCategory.FACTUALITY`: How factual a response is
- `ThresholdCategory.HALLUCINATION`: Detection threshold for hallucinations
- `ThresholdCategory.CONFIDENCE`: Confidence score threshold
- `ThresholdCategory.CONSTRAINT`: Constraint validation threshold
- `ThresholdCategory.LATENCY`: Maximum acceptable latency (in ms)
- `ThresholdCategory.RETRY`: Threshold for retry decisions
- `ThresholdCategory.FALLBACK`: Threshold for fallback decisions
- `ThresholdCategory.COMBINED`: Combined threshold for overall assessment

## Implementation Details

### Default Values

The default threshold values are defined in the `DEFAULT_THRESHOLDS` dictionary in `threshold_unified.py`. These values represent the baseline for production mode.

### Mode Adjustments

The `MODE_ADJUSTMENTS` dictionary contains multipliers for each threshold category in different execution modes:

- **Production**: No adjustments (multiplier = 1.0)
- **Benchmark**: Slightly more lenient (multiplier = 0.95 for most categories)
- **Development**: More lenient (multiplier = 0.9 for most categories)
- **Testing**: Most lenient (multiplier = 0.8 for most categories)

For latency thresholds, the multiplier is above 1.0 since higher values are more lenient (allowing more time).

## Migration Guide

To migrate existing code to use the unified threshold system:

1. Replace domain-specific threshold imports with the unified system:
   ```python
   # Old approach
   from config.domain_config import Domain, get_threshold
   
   # New approach
   from config.domain_config import Domain
   from config.threshold_unified import get_threshold, ThresholdCategory
   ```

2. Replace threshold retrievals:
   ```python
   # Old approach
   threshold = get_threshold(domain, "factuality")
   
   # New approach
   threshold = get_threshold(domain, ThresholdCategory.FACTUALITY)
   ```

3. Set the execution mode appropriately:
   ```python
   # For benchmark code
   from config.threshold_unified import set_execution_mode, ExecutionMode
   set_execution_mode(ExecutionMode.BENCHMARK)
   
   # For production code
   set_execution_mode(ExecutionMode.PRODUCTION)
   ```

4. Use the automated migration script to update your files:
   ```bash
   python config/migrate_thresholds.py --backup
   ```

## Compatibility

The unified threshold system includes compatibility layers for existing code:

1. **Domain Config Compatibility**: The function `get_legacy_domain_threshold` provides the same interface as the original `get_threshold` function in `domain_config.py`

2. **Grounding Engine Compatibility**: The constants `FACTUALITY_THRESHOLD` and `DOMAIN_THRESHOLDS` in `grounding_engine.py` are maintained for backward compatibility

3. **Benchmark Compatibility**: The `ThresholdConfig` class in `threshold_config.py` is updated to use the unified system internally while maintaining its original interface

## Configuration Management

The unified threshold system supports exporting and importing configurations:

```python
from config.threshold_unified import export_configuration, import_configuration

# Export current configuration
export_configuration("/path/to/thresholds.yml")

# Import configuration
import_configuration("/path/to/thresholds.yml")
```

## Troubleshooting

If you encounter issues with the unified threshold system:

1. **Verify Execution Mode**: Check that the correct execution mode is set for your environment
2. **Check for Overrides**: Runtime overrides may be affecting threshold values
3. **Inspect Logs**: The threshold system logs information about its operations
4. **Reset to Defaults**: Use `clear_threshold_overrides()` to reset any custom overrides

## Best Practices

1. **Always Use Enum Values**: Prefer `ThresholdCategory.FACTUALITY` over string literals like `"factuality"`
2. **Set Mode Explicitly**: At the start of your script or application, explicitly set the execution mode
3. **Use Domain Enums**: Prefer `Domain.MEDICAL` over string literals like `"medical"`
4. **Verify Thresholds**: When debugging, print threshold values to verify they match expectations
5. **Document Overrides**: When using threshold overrides, document the reason and expected duration

## Further Reading

- [Domain Configuration Guide](domain_config.md)
- [Benchmark Performance Framework](../docs/performance_framework.md)
- [Grounding System Documentation](../v3/grounding/README.md)