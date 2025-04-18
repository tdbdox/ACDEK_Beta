 # ACDEK v3 SDK Guide

 ## Overview

 The ACDEK SDK provides a simple `Client` class to interact with core v3 components:

 - **Enrichment Pipeline**: Data enrichment stages via plugins
 - **Constraint Verifier**: Formal constraint validations
 - **Confidence Scorer**: Evidence-based confidence metrics

 ## Installation

 ```bash
 pip install -e .  # from the v3 directory
 ```

 ## Importing the Client

 ```python
 from sdk.acdek_client import Client
 ```

 ## Creating a Client

 ```python
 config = {
     "fallback_mode": "strict",      # Options: strict, lenient, deferred
     "constraint_plugins": None,      # List of plugin names or None for all
 }
 client = Client(config=config)
 ```

 ## Client Methods

 ### `run_enrichment(input_data, context=None, output_format="dict")`
 - Executes the enrichment pipeline.

 ### `validate_constraints(input_data, output_format="dict")`
 - Validates data against registered constraints.

 ### `score_confidence(input_data, output_format="dict")`
 - Computes confidence score with factor breakdown.

 ### `full_pipeline(input_data, context=None, output_format="dict")`
 - Runs enrichment, constraint validation, and confidence scoring sequentially.

 ## Configuration Options

 - **`fallback_mode`**: Controls strictness of errors (strict, lenient, deferred).
 - **`constraint_plugins`**: List of constraint plugin names or all by default.

 ## Common Gotchas

 - Ensure input payload matches plugin expectations (e.g., `vector`, `model_name`).
 - If no constraint plugins are registered, validation passes by default.

 ## Output Explanation

 - **Enrichment Result**: `success`, `data`, `errors`, `metrics`, `stage_results`.
 - **Constraint Result**: `passed`, `results`, `violations`, `metrics`.
 - **Confidence Result**: `aggregate_score`, `factors`, `metadata`, `timestamp`.