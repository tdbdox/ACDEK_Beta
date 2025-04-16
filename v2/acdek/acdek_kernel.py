from acdek.solve_Q import solve_Q
from acdek.solve_Ar_vector import solve_Ar
from acdek.solve_Cu import solve_Cu
from acdek.solve_ef import solve_ef
from acdek.gamma import gamma
from acdek.gate import gate
from acdek.phi import phi
from acdek.omega import omega
from acdek.types import CuRaw, EfRaw, QRaw, Theta
import logging
import time
import traceback
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_acdek_pipeline(Q_raw: Union[str, QRaw], Ar_raw: dict, Cu_raw: CuRaw, ef_raw: EfRaw, theta: Union[dict, Theta] = None) -> dict:
    """
    Execute the full ACDEK pipeline with improved error handling and diagnostics.
    
    Args:
        Q_raw: Raw query string or QRaw object
        Ar_raw: Raw data sources
        Cu_raw: Raw user context
        ef_raw: Raw environment factors
        theta: Optional modulation vector override (dict or Theta object)
        
    Returns:
        Dictionary containing the final result or fallback information
    """
    start_time = time.time()
    pipeline_metrics = {"stages": {}}
    result = {"status": "fallback_triggered", "reason": "Unknown error in pipeline"}
    
    try:
        # Wrap Q_raw in QRaw if it's a string
        if isinstance(Q_raw, str):
            Q_raw = QRaw(query=Q_raw)
            
        logger.info(f"Starting ACDEK pipeline processing for query: {Q_raw.query if hasattr(Q_raw, 'query') else str(Q_raw)[:50]}")
        
        # Step 1: Enrich Query
        stage_start = time.time()
        try:
            Q_enriched = solve_Q(Q_raw, Cu_raw, ef_raw)
            logger.info(f"Q_enriched.intent (semantic): {Q_enriched.intent}")
            logger.info(f"Q_enriched.dependencies (vector-matched): {Q_enriched.dependencies}")
            logger.info(f"Q_enriched.ambiguity_score: {Q_enriched.ambiguity_score}")
            pipeline_metrics["stages"]["Q"] = {"success": True, "time": time.time() - stage_start}
        except Exception as e:
            logger.error(f"Error in solve_Q: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "component": "solve_Q",
                "error": str(e),
                "error_trace": traceback.format_exc(),
                "metrics": pipeline_metrics
            }

        # Step 2: Enrich Data
        stage_start = time.time()
        try:
            Ar_enriched = solve_Ar(Ar_raw, Q_enriched)
            logger.info(f"Ar_enriched.slots (semantic): {list(Ar_enriched.slots.keys())}")
            for k, v in Ar_enriched.slots.items():
                logger.info(f"     ↳ {k} → value: {v.value}, confidence: {v.confidence:.2f}")
            pipeline_metrics["stages"]["Ar"] = {"success": True, "time": time.time() - stage_start}
        except Exception as e:
            logger.error(f"Error in solve_Ar: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "component": "solve_Ar",
                "error": str(e),
                "error_trace": traceback.format_exc(),
                "metrics": pipeline_metrics,
                "query_info": {
                    "intent": Q_enriched.intent,
                    "dependencies": Q_enriched.dependencies,
                    "ambiguity_score": Q_enriched.ambiguity_score
                }
            }

        # Step 3: Enrich User Context
        stage_start = time.time()
        try:
            Cu_enriched = solve_Cu(Cu_raw, Q_enriched, Ar_enriched)
            
            if isinstance(Cu_enriched, dict) and Cu_enriched.get("semantic_class"):
                logger.info(f"Matched semantic class: {Cu_enriched['semantic_class']}")
                logger.info(f"Confidence: {Cu_enriched['confidence']:.4f}")
                logger.info(f"Status: {Cu_enriched.get('status', 'not_specified')}")
            pipeline_metrics["stages"]["Cu"] = {
                "success": True, 
                "time": time.time() - stage_start,
                "status": Cu_enriched.get("status")
            }
        except Exception as e:
            logger.error(f"Error in solve_Cu: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a minimal Cu_enriched to allow pipeline to continue
            Cu_enriched = {"semantic_class": None, "object": {}, "confidence": 0.0, "error": str(e)}
            pipeline_metrics["stages"]["Cu"] = {"success": False, "time": time.time() - stage_start, "error": str(e)}

        # Step 4: Enrich Environment
        stage_start = time.time()
        try:
            ef_enriched = solve_ef(ef_raw, Q_enriched, Cu_enriched)
            pipeline_metrics["stages"]["ef"] = {"success": True, "time": time.time() - stage_start}
        except Exception as e:
            logger.error(f"Error in solve_ef: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a minimal ef_enriched to allow pipeline to continue
            ef_enriched = {}
            pipeline_metrics["stages"]["ef"] = {"success": False, "time": time.time() - stage_start, "error": str(e)}

        # Step 5: Modulation Vector
        stage_start = time.time()
        try:
            theta = omega(Cu_enriched, ef_enriched, override=theta)
            pipeline_metrics["stages"]["omega"] = {"success": True, "time": time.time() - stage_start}
        except Exception as e:
            logger.error(f"Error in omega: {str(e)}")
            logger.error(traceback.format_exc())
            # Use default theta values to allow pipeline to continue
            theta = {
                "tone": 0.5,
                "assertiveness": 0.5,
                "confidence_floor": 0.7,
                "fallback_policy": "clarify"
            }
            pipeline_metrics["stages"]["omega"] = {"success": False, "time": time.time() - stage_start, "error": str(e)}

        # Step 6: Verification / Payload Formation
        stage_start = time.time()
        try:
            payload_result = gamma(Q_enriched, Ar_enriched, theta)
            pipeline_metrics["stages"]["gamma"] = {"success": True, "time": time.time() - stage_start}
        except Exception as e:
            logger.error(f"Error in gamma: {str(e)}")
            logger.error(traceback.format_exc())
            # Return graceful fallback since gamma failed
            pipeline_metrics["stages"]["gamma"] = {"success": False, "time": time.time() - stage_start, "error": str(e)}
            pipeline_metrics["total_time"] = time.time() - start_time
            return {
                "status": "fallback_triggered",
                "reason": f"Error in verification: {str(e)}",
                "justification": "Returning safe fallback due to verification error.",
                "modulation": theta,
                "gate_decision": "fallback",
                "ambiguity_score": Q_enriched.ambiguity_score,
                "metrics": pipeline_metrics,
                "dependencies": Q_enriched.dependencies
            }

        if "fallback" in payload_result:
            pipeline_metrics["total_time"] = time.time() - start_time
            return {
                "status": "fallback_triggered",
                "reason": payload_result.get("reason", "Unknown failure"),
                "details": payload_result.get("details", {}),
                "justification": "Returning safe fallback due to unsatisfied conditions.",
                "modulation": theta,
                "gate_decision": "fallback",
                "ambiguity_score": Q_enriched.ambiguity_score,
                "metrics": pipeline_metrics,
                "dependencies": Q_enriched.dependencies
            }

        # Unwrap payload
        payload = payload_result["payload"]
        
        # Add semantic class information if available
        if isinstance(Cu_enriched, dict) and Cu_enriched.get("semantic_class"):
            payload["user_class"] = Cu_enriched["semantic_class"]
            payload["user_class_confidence"] = Cu_enriched["confidence"]
            payload["semantic_class"] = Cu_enriched["semantic_class"]
            payload["semantic_class_confidence"] = Cu_enriched["confidence"]
            if "coverage" in Cu_enriched:
                payload["user_class_coverage"] = Cu_enriched["coverage"]
                payload["trait_coverage"] = Cu_enriched["coverage"]

        logger.info(f"Final payload keys before gate: {payload.keys()}")
        logger.info(f"Confidence values before gate: {payload.get('confidence', {})}")

        payload["ambiguity_score"] = Q_enriched.ambiguity_score
        payload["dependencies"] = Q_enriched.dependencies

        # Step 7: Gating Decision
        stage_start = time.time()
        try:
            decision = gate(payload, theta)
            logger.info(f"Gate decision: {decision}")
            pipeline_metrics["stages"]["gate"] = {"success": True, "time": time.time() - stage_start, "decision": decision}
        except Exception as e:
            logger.error(f"Error in gate: {str(e)}")
            logger.error(traceback.format_exc())
            # Default to fallback on gate error
            decision = "fallback"
            pipeline_metrics["stages"]["gate"] = {"success": False, "time": time.time() - stage_start, "error": str(e)}

        # Step 8: Final Execution Path
        stage_start = time.time()
        try:
            # Add the semantic_class and theta to phi's input to enable proper memory-trace conditioning
            if isinstance(Cu_enriched, dict) and Cu_enriched.get("semantic_class"):
                # Ensure phi has access to the semantic_class for memory conditioning
                if "semantic_class" not in payload:
                    payload["semantic_class"] = Cu_enriched["semantic_class"]
                    
                # Add traits if needed for memory conditioning
                if "traits" in Cu_enriched and "traits" not in payload:
                    payload["traits"] = Cu_enriched["traits"]
            
            result = phi(decision, payload, theta)
            pipeline_metrics["stages"]["phi"] = {"success": True, "time": time.time() - stage_start}
        except Exception as e:
            logger.error(f"Error in phi: {str(e)}")
            logger.error(traceback.format_exc())
            pipeline_metrics["stages"]["phi"] = {"success": False, "time": time.time() - stage_start, "error": str(e)}
            # Return minimal fallback information
            result = {
                "status": "fallback_triggered",
                "reason": f"Error in execution: {str(e)}",
                "justification": "Returning safe fallback due to execution error.",
                "modulation": theta,
                "gate_decision": decision,
                "ambiguity_score": Q_enriched.ambiguity_score,
                "dependencies": Q_enriched.dependencies
            }

        # Add pipeline metrics to result
        pipeline_metrics["total_time"] = time.time() - start_time
        result["metrics"] = pipeline_metrics
        logger.info(f"ACDEK pipeline completed in {pipeline_metrics['total_time']:.4f} seconds with status: {result['status']}")
        
        return result
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected error in ACDEK pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        
        total_time = time.time() - start_time
        pipeline_metrics["total_time"] = total_time
        
        return {
            "status": "error",
            "reason": f"Unexpected pipeline error: {str(e)}",
            "error_trace": traceback.format_exc(),
            "metrics": pipeline_metrics
        }
