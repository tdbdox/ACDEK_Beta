#!/usr/bin/env python3
"""
ACDEK Pipeline CLI.
"""
import sys
import json
import argparse
import logging
import time

from sdk.acdek_client import Client

def main():
    parser = argparse.ArgumentParser(description="Run ACDEK pipeline with SDK or start an interactive session.")
    parser.add_argument(
        "-i", "--input", type=str,
        help="Input JSON file path (default: stdin)"
    )
    parser.add_argument(
        "-f", "--fallback-mode", type=str, default="strict",
        choices=["strict", "lenient", "deferred"],
        help="Fallback mode"
    )
    parser.add_argument(
        "-l", "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "-o", "--output-format", type=str, default="pretty",
        choices=["pretty", "json"],
        help="Output format"
    )
    parser.add_argument(
        "--session", action="store_true",
        help="Enable interactive session mode (REPL style)."
    )
    parser.add_argument(
        "--export-session", type=str, default=None,
        help="Path to JSON file to export session metrics upon exit."
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Load input data
    # Session mode or single run
    if args.session:
        # Interactive session
        from acdek.acdek_kernel import ACDEKKernel
        from observability.metrics_collector import metrics_collector
        kernel = ACDEKKernel(config={"fallback_mode": args.fallback_mode})
        session_stats = {"runs": 0, "total_time": 0.0, "fallback_depth": 0, "fallback_penalty": 0.0}
        print("Entering ACDEK interactive session. Type JSON input per line, Ctrl-D to exit.")
        while True:
            try:
                line = input("acdek> ")
            except EOFError:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                logging.error(f"Invalid JSON input: {e}")
                continue
            # Run pipeline
            start = time.time()
            result = kernel.run(data)
            duration = time.time() - start
            # Record session metrics
            session_stats["runs"] += 1
            session_stats["total_time"] += duration
            ed = result.get("enrichment", {})
            session_stats["fallback_depth"] += ed.get("fallback_depth", 0)
            session_stats["fallback_penalty"] += ed.get("fallback_penalty", 0.0)
            # Record observability metrics
            metrics_collector.record_request(duration)
            timings = result.get("timing", {})
            for stage in ("enrichment", "constraint", "confidence"):
                if stage in timings:
                    metrics_collector.record_stage_invocation(stage)
                    metrics_collector.record_stage_duration(stage, timings[stage])
            # Output result
            if args.output_format == "json":
                print(json.dumps(result, indent=2))
            else:
                # reuse pretty output
                enrich = result.get("enrichment", {})
                print("== Enrichment ==")
                print(f"Success: {enrich.get('success')}")
                errors = enrich.get('errors')
                print(f"Errors: {errors if errors else 'None'}")
                constr = result.get("constraints", {})
                print("\n== Constraints ==")
                print(f"Passed: {constr.get('passed')}")
                violations = constr.get('violations')
                if violations:
                    print("Violations:")
                    for v in violations:
                        print(f" - {v.get('name')}: {v.get('message')}")
                else:
                    print("No violations")
                conf = result.get("confidence", {})
                print("\n== Confidence ==")
                agg = conf.get('aggregate_score')
                try:
                    print(f"Aggregate Score: {agg:.3f}")
                except Exception:
                    print(f"Aggregate Score: {agg}")
                factors = conf.get('factors', {})
                print("Factors:")
                for k, v in factors.items():
                    try:
                        print(f" - {k}: {v:.3f}")
                    except Exception:
                        print(f" - {k}: {v}")
                print(f"\nFallback Mode: {args.fallback_mode}")
        # Session end
        if args.export_session:
            try:
                with open(args.export_session, 'w') as ef:
                    import json as _json
                    export = {"session_stats": session_stats, "metrics": metrics_collector.snapshot()}
                    _json.dump(export, ef, indent=2)
                print(f"Session metrics exported to {args.export_session}")
            except Exception as e:
                logging.error(f"Failed to export session metrics: {e}")
        return
    # Single-run mode
    try:
        if args.input:
            with open(args.input, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)
    except Exception as e:
        logging.error(f"Failed to load input data: {e}")
        sys.exit(1)

    # Initialize client for single run
    client = Client(config={"fallback_mode": args.fallback_mode})
    try:
        result = client.full_pipeline(data, output_format="dict")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

    # Output results
    if args.output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        # Pretty output
        enrich = result.get("enrichment", {})
        print("== Enrichment ==")
        print(f"Success: {enrich.get('success')}")
        errors = enrich.get('errors')
        print(f"Errors: {errors if errors else 'None'}")

        constr = result.get("constraints", {})
        print("\n== Constraints ==")
        print(f"Passed: {constr.get('passed')}")
        violations = constr.get('violations')
        if violations:
            print("Violations:")
            for v in violations:
                print(f" - {v.get('name')}: {v.get('message')}")
        else:
            print("No violations")

        conf = result.get("confidence", {})
        print("\n== Confidence ==")
        agg = conf.get('aggregate_score')
        try:
            print(f"Aggregate Score: {agg:.3f}")
        except Exception:
            print(f"Aggregate Score: {agg}")
        factors = conf.get('factors', {})
        print("Factors:")
        for k, v in factors.items():
            try:
                print(f" - {k}: {v:.3f}")
            except Exception:
                print(f" - {k}: {v}")

        print(f"\nFallback Mode: {args.fallback_mode}")

if __name__ == '__main__':
    main()