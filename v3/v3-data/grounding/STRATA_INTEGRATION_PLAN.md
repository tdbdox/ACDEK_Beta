# Strata Integration Plan for ACDEK v3 Phase 3 Grounding Improvements

## Overview

This document outlines the plan to integrate the ACDEK v3 Phase 3 grounding improvements into the Strata platform. The integration will focus on ensuring the enhanced grounding capabilities are available to Strata while maintaining backward compatibility and service stability.

## Current Integration Structure

The Strata backend integrates ACDEK through:

1. **Docker Deployment**: The `acdek-kernel.Dockerfile` copies the ACDEK v3 code from a local path and installs it in development mode.

2. **Kernel Service**: The `kernel_service.py` creates and manages ACDEK kernel instances for different tenants.

3. **API Integration**: Various Strata components (market strategy, reporting) import and use ACDEK functionality through integration modules.

## Phase 3 Improvements to Integrate

The key improvements from Phase 3 to integrate include:

1. **Enhanced Grounding Engine**: The `grounding_engine_enhanced.py` with:
   - Adaptive thresholds
   - Domain-specific factuality thresholds
   - Performance optimizations

2. **Enhanced Reference Data Structure**: The structured reference data format that improves hallucination detection.

3. **Performance Optimizations**: The 42x performance improvement provided by adaptive thresholds.

## Integration Steps

### 1. Code Synchronization

Ensure the latest version of the Phase 3 implementation is synchronized with the version that will be used by Strata:

```bash
# Run this from the strata-backend directory
mkdir -p deploy/temp/acdek_v3_phase3
cp -r /Users/travispatchett/Solvion/Development/ACDEK_BETA/v3/v3-data/grounding/*_enhanced.py deploy/temp/acdek_v3_phase3/
cp -r /Users/travispatchett/Solvion/Development/ACDEK_BETA/v3/v3-data/data/enhanced_* deploy/temp/acdek_v3_phase3/
```

### 2. Docker Image Update

Update the Dockerfile to include the Phase 3 improvements:

```dockerfile
# Modified part of the Dockerfile
COPY strata-backend/deploy/temp/acdek_v3_phase3/ /app/acdek/v3-data/grounding/
```

### 3. Configuration Updates

Create a configuration update that enables the Phase 3 features by default:

```python
# Add to kernel_service.py in the _get_tenant_config method
default_config.update({
    "grounding": {
        "use_enhanced": True,
        "use_adaptive_threshold": True,
        "reference_data_path": "/app/acdek/v3-data/data/enhanced_{domain}"
    }
})
```

### 4. ACDEK Kernel Integration Point

Ensure the ACDEK kernel is configured to use the enhanced grounding components:

```python
# Import the enhanced components
try:
    from grounding_engine_enhanced import (
        grounded_validation_pipeline_enhanced as validate_grounding,
        adaptive_threshold
    )
    ENHANCED_GROUNDING_AVAILABLE = True
except ImportError:
    # Fallback to standard components
    from grounding_engine import grounded_validation_pipeline as validate_grounding
    ENHANCED_GROUNDING_AVAILABLE = False
```

### 5. Testing Strategy

1. **Unit Testing**: Create unit tests for the enhanced components.
2. **Integration Testing**: Test the integration between Strata and the enhanced ACDEK components.
3. **Performance Benchmarking**: Compare the performance of the original and enhanced components.
4. **Regression Testing**: Ensure existing functionality continues to work as expected.

### 6. Deployment Strategy

1. **Canary Release**: Initially deploy to a small subset of tenants.
2. **Feature Flag**: Use a feature flag to enable or disable enhanced components.
3. **Rollback Plan**: Prepare a rollback plan in case of unexpected issues.
4. **Monitoring**: Implement detailed monitoring to track performance and errors.

## Implementation Details

### Enhanced Grounding Engine Integration

1. Update `enrichment_pipeline.py` to include the enhanced components:

```python
def create_enhanced_pipeline(config=None):
    """
    Create an enhanced pipeline with Phase 3 improvements.
    """
    if config is None:
        config = {}
    
    pipeline = EnrichmentPipeline(config)
    
    # Add standard components
    pipeline.add(ConfidenceScorer(config.get("confidence", {})))
    pipeline.add(ConstraintVerifier(config.get("constraints", {})))
    
    # Add enhanced grounding components
    use_adaptive = config.get("grounding", {}).get("use_adaptive_threshold", True)
    
    pipeline.add(GroundingEngineEnhanced(
        config.get("grounding", {}),
        use_adaptive_threshold=use_adaptive
    ))
    
    return pipeline
```

2. Update the kernel initialization to use the enhanced pipeline:

```python
def create_kernel(tenant_id, config=None):
    """
    Create a kernel instance with the appropriate configuration.
    """
    effective_config = _get_tenant_config(tenant_id)
    if config:
        effective_config.update(config)
    
    # Determine whether to use enhanced components
    use_enhanced = effective_config.get("grounding", {}).get("use_enhanced", True)
    
    if use_enhanced and ENHANCED_GROUNDING_AVAILABLE:
        pipeline = create_enhanced_pipeline(effective_config)
    else:
        pipeline = create_standard_pipeline(effective_config)
    
    kernel = ACDEKKernel(config=effective_config)
    kernel.set_pipeline(pipeline)
    
    return kernel
```

### Performance Monitoring

Add specific metrics to track the performance of the enhanced components:

```python
# In kernel_service.py

def _register_metrics(self):
    """Register metrics for kernel registry."""
    # Existing metrics...
    
    # Add grounding-specific metrics
    self.metrics.register_histogram(
        "grounding.processing_time",
        "Grounding processing time in milliseconds",
        buckets=[1, 5, 10, 50, 100, 500]
    )
    
    self.metrics.register_counter(
        "grounding.hallucinations_detected",
        "Number of hallucinations detected"
    )
    
    self.metrics.register_gauge(
        "grounding.false_positive_rate",
        "Estimated false positive rate"
    )
```

## Risk Mitigation

1. **Performance Degradation**: Monitor processing time closely and be prepared to roll back if performance degrades.

2. **Accuracy Issues**: Track false positive and false negative rates to ensure accuracy meets expectations.

3. **Dependency Conflicts**: Verify all dependencies are compatible between Strata and the enhanced ACDEK components.

4. **Resource Utilization**: Monitor memory and CPU usage to ensure the enhanced components don't cause resource constraints.

## Success Criteria

1. **Performance**: Processing time reduced by at least 10x compared to the baseline.

2. **Stability**: No increase in error rates or service disruptions.

3. **Accuracy**: Maintain high recall while reducing false positives where possible.

4. **User Experience**: Improved response times and more accurate hallucination detection in Strata applications.

## Timeline

1. **Code Integration**: 1 day
2. **Testing**: 2 days
3. **Canary Deployment**: 1 day
4. **Full Deployment**: 1 day
5. **Monitoring and Adjustment**: Ongoing

## Conclusion

The integration of ACDEK v3 Phase 3 grounding improvements into Strata will significantly enhance the platform's capability to detect hallucinations while dramatically improving performance. By following this integration plan, we can ensure a smooth transition to the enhanced components while minimizing risks and disruptions to existing functionality.