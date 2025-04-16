from acdek.acdek_kernel import run_acdek_pipeline
from acdek.types import CuRaw, EfRaw

# Define mock query
Q_raw = "Generate sales summary report"

# Define mock data input (Ar_raw)
Ar_raw = {
    "sales_data": {"value": "Q1 sales figures", "age": 3},
    "report_format": {"value": "pdf", "age": 1}
}

# Define mock user context
Cu_raw = CuRaw(traits={
    "role": "analyst",
    "risk_tolerance": "medium"
})

# Define mock environmental context
ef_raw = EfRaw(variables={
    "season": "spring",
    "urgency": "high",
    "volatility": "low"
})

# Run the full ACDEK pipeline
result = run_acdek_pipeline(Q_raw, Ar_raw, Cu_raw, ef_raw)

# Output the result
print("=== ACDEK Kernel Demo Result ===")
for key, value in result.items():
    print(f"{key}: {value}")
