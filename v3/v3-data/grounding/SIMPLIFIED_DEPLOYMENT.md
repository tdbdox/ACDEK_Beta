# Simplified Deployment Strategy for ACDEK v3 Phase 3

## Overview

This document provides a simplified deployment strategy for integrating the ACDEK v3 Phase 3 enhancements into the Strata platform. This approach avoids complex Docker rebuilds and focuses on direct code integration.

## Step 1: Copy Enhanced Components

The enhanced components have already been copied to the Strata deployment directory:
```
/Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/
```

## Step 2: Integrate with Market Strategy Module

1. Copy the integration adapter to the market strategy directory:

```bash
cp /Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/acdek_integration_enhanced.py /Users/travispatchett/Solvion/Development/strata-backend/src/workflow_engine/market_strategy/
```

2. Update the market strategy integration file:

```bash
# Edit the file
nano /Users/travispatchett/Solvion/Development/strata-backend/src/workflow_engine/market_strategy/acdek_integration.py
```

Add these lines at the top of the file:
```python
try:
    from .acdek_integration_enhanced import get_grounding_adapter
    ENHANCED_GROUNDING_AVAILABLE = True
except ImportError:
    ENHANCED_GROUNDING_AVAILABLE = False
```

3. Modify the validation functions to use the enhanced adapter:

```python
def validate_strategy(strategy):
    """Validate a market strategy using ACDEK."""
    if ENHANCED_GROUNDING_AVAILABLE:
        # Use enhanced grounding
        adapter = get_grounding_adapter()
        validation_result = adapter.validate_content(
            prompt=strategy.description,
            response=strategy.implementation_details,
            domain="general"
        )
        return validation_result
    else:
        # Use original validation
        # ... existing code ...
```

## Step 3: Integrate with Reporting Module

1. Copy the integration adapter to the reporting directory:

```bash
cp /Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/acdek_integration_enhanced.py /Users/travispatchett/Solvion/Development/strata-backend/src/workflow_engine/reporting/
```

2. Update the reporting integration file:

```bash
# Edit the file
nano /Users/travispatchett/Solvion/Development/strata-backend/src/workflow_engine/reporting/acdek_integration.py
```

Add these lines at the top of the file:
```python
try:
    from .acdek_integration_enhanced import get_grounding_adapter
    ENHANCED_GROUNDING_AVAILABLE = True
except ImportError:
    ENHANCED_GROUNDING_AVAILABLE = False
```

3. Modify the validation functions to use the enhanced adapter.

## Step 4: Update the ACDEK Kernel Service

1. Copy the enhanced grounding components to the ACDEK module directory:

```bash
# Create the directory if it doesn't exist
mkdir -p /Users/travispatchett/Solvion/Development/strata-backend/src/acdek_kernel/enhanced_grounding

# Copy the files
cp /Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/*.py /Users/travispatchett/Solvion/Development/strata-backend/src/acdek_kernel/enhanced_grounding/
```

2. Create an integration module for the ACDEK kernel service:

```bash
# Create the integration file
nano /Users/travispatchett/Solvion/Development/strata-backend/src/acdek_kernel/enhanced_grounding_integration.py
```

Add this content:
```python
"""
Integration module for the enhanced grounding components.
"""

import os
import sys
from pathlib import Path

# Add the enhanced grounding directory to the Python path
current_dir = Path(__file__).resolve().parent
enhanced_grounding_dir = current_dir / "enhanced_grounding"
sys.path.append(str(enhanced_grounding_dir))

try:
    from acdek_integration_enhanced import get_grounding_adapter
    ENHANCED_GROUNDING_AVAILABLE = True
except ImportError:
    ENHANCED_GROUNDING_AVAILABLE = False

def get_enhanced_grounding_adapter(**kwargs):
    """
    Get the enhanced grounding adapter with the given configuration.
    
    Returns:
        The enhanced grounding adapter, or None if not available.
    """
    if ENHANCED_GROUNDING_AVAILABLE:
        return get_grounding_adapter(kwargs)
    return None
```

3. Update the kernel service to use the enhanced grounding if available:

```bash
# Edit the kernel service
nano /Users/travispatchett/Solvion/Development/strata-backend/src/acdek_kernel/kernel_service.py
```

Add this import at the top:
```python
try:
    from .enhanced_grounding_integration import get_enhanced_grounding_adapter, ENHANCED_GROUNDING_AVAILABLE
except ImportError:
    ENHANCED_GROUNDING_AVAILABLE = False
    get_enhanced_grounding_adapter = None
```

## Step 5: Create Data Directories

1. Create directories for the enhanced reference data:

```bash
# Create directories
mkdir -p /Users/travispatchett/Solvion/Development/strata-backend/data/enhanced_medical
mkdir -p /Users/travispatchett/Solvion/Development/strata-backend/data/enhanced_legal
mkdir -p /Users/travispatchett/Solvion/Development/strata-backend/data/enhanced_finance

# Copy reference data
cp /Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/data/enhanced_medical/* /Users/travispatchett/Solvion/Development/strata-backend/data/enhanced_medical/
cp /Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/data/enhanced_legal/* /Users/travispatchett/Solvion/Development/strata-backend/data/enhanced_legal/
cp /Users/travispatchett/Solvion/Development/strata-backend/deploy/temp/acdek_v3_phase3/data/enhanced_finance/* /Users/travispatchett/Solvion/Development/strata-backend/data/enhanced_finance/
```

## Step 6: Restart the Services

1. Restart the web server to apply the changes:

```bash
cd /Users/travispatchett/Solvion/Development/strata-backend
./run_server.sh
```

## Deployment Verification

To verify that the enhanced grounding is working:

1. Check the logs for "Enhanced grounding components loaded" messages:

```bash
grep "Enhanced grounding" /Users/travispatchett/Solvion/Development/strata-backend/strata.log
```

2. Test the market strategy validation:

```bash
cd /Users/travispatchett/Solvion/Development/strata-backend
python3 test_market_strategy_api.py
```

3. Monitor performance metrics to confirm the improved processing time.

## Rollback Plan

If any issues arise, you can roll back by:

1. Removing the enhanced integration files:

```bash
rm /Users/travispatchett/Solvion/Development/strata-backend/src/workflow_engine/market_strategy/acdek_integration_enhanced.py
rm /Users/travispatchett/Solvion/Development/strata-backend/src/workflow_engine/reporting/acdek_integration_enhanced.py
rm -rf /Users/travispatchett/Solvion/Development/strata-backend/src/acdek_kernel/enhanced_grounding
```

2. Reverting the changes to the integration files.

3. Restarting the services.

## Conclusion

This simplified deployment approach avoids the complexity of Docker rebuilds while still integrating the enhanced ACDEK v3 Phase 3 components. The graceful fallback mechanisms ensure that the system will continue to function even if the enhanced components are unavailable, making this a safe deployment strategy.