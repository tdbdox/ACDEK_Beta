 # ACDEK v3 Pipeline Example

 This example demonstrates the end-to-end ACDEK pipeline, including enrichment, constraint validation, and confidence scoring.

 ## Sample Input

 ```json
 {
   "text": "Assess the current market trends and provide an analysis."
 }
 ```

 ## SDK Invocation

 ```python
 from sdk.acdek_client import Client

 client = Client(config={"fallback_mode": "lenient"})
 result = client.full_pipeline({"text": "Assess the current market trends."})
 print(result)
 ```

 ## Sample Output

 ```json
 {
   "enrichment": {
     "success": true,
     "data": { /* enriched fields */ },
     "errors": {},
     "metrics": { /* timing */ },
     "stage_results": { /* per-stage output */ }
   },
   "constraints": {
     "passed": true,
     "results": [],
     "violations": [],
     "metrics": { "duration": 0.01, "total_checks": 0 }
   },
   "confidence": {
     "aggregate_score": 0.84,
     "factors": { /* scores */ },
     "metadata": { /* uncertainty, decay */ },
     "timestamp": 1610000000.0
   }
 }
 ```

 ## Interpretation

 - **Enrichment**: The pipeline added new slots/embeddings to `data`.
 - **Constraints**: All formal checks passed; no violations.
 - **Confidence**: Aggregate score indicates reliability of enriched data.

 For more details, refer to `docs/sdk_guide.md` and `docs/pipeline_example.md`.