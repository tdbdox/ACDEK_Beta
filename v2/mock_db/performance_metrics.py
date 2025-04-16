# mock_db/performance_metrics.py

performance_metrics = {
    "sales_performance": {
        "value": "Q2 sales increased by 12%",
        "confidence": 0.87,
        "timestamp": "2025-04-01"
    },
    "engineering_output": {
        "value": "Deployment frequency improved",
        "confidence": 0.81,
        "timestamp": "2025-04-02"
    },
    "team_engagement": {
        "value": "Engagement up by 15%",
        "confidence": 0.92,
        "timestamp": "2025-04-03"
    },
    "burnout_index": {
        "value": "Slight increase in stress reports",
        "confidence": 0.66,
        "timestamp": "2025-04-04"
    }
}

# Detailed sales data for summarize_sales intent
sales_data = {
    "q2_revenue": {
        "value": "$2.4M in Q2 revenue, up 12% from Q1",
        "confidence": 0.95,
        "timestamp": "2025-04-01"
    },
    "sales_by_region": {
        "value": "North: +15%, East: +8%, South: +5%, West: +18%",
        "confidence": 0.92,
        "timestamp": "2025-04-01"
    },
    "top_products": {
        "value": "Premium subscription showing 22% growth",
        "confidence": 0.91,
        "timestamp": "2025-04-01"
    },
    "customer_acquisition": {
        "value": "New customers up 7% with improved retention",
        "confidence": 0.89,
        "timestamp": "2025-04-02"
    },
    "quarterly_comparison": {
        "value": "Three consecutive quarters of growth",
        "confidence": 0.94,
        "timestamp": "2025-04-01" 
    }
}

# Inventory and trend data for track_inventory intent
trend_indicators = {
    "inventory_levels": {
        "value": "Current stock at 85% capacity, optimal range",
        "confidence": 0.93,
        "timestamp": "2025-04-05"
    },
    "stock_rotation": {
        "value": "Inventory turnover improved to 4.2x quarterly",
        "confidence": 0.88,
        "timestamp": "2025-04-05"
    },
    "historical_trend": {
        "value": "15% reduction in overstock compared to Q1",
        "confidence": 0.90,
        "timestamp": "2025-04-05"
    },
    "reorder_optimization": {
        "value": "Just-in-time ordering resulted in 8% cost reduction",
        "confidence": 0.85,
        "timestamp": "2025-04-05"
    },
    "quarter_trend": {
        "value": "Inventory efficiency improved over last 3 quarters",
        "confidence": 0.92,
        "timestamp": "2025-04-05" 
    }
}

# Quarterly comparison data specifically for compare_quarters intent
quarter_comparison_data = {
    "q1_to_q2_growth": {
        "value": "Q2 showed 12% revenue growth compared to Q1",
        "confidence": 0.97,
        "timestamp": "2025-04-03"
    },
    "q2_inventory_vs_q1": {
        "value": "Inventory efficiency in Q2 improved 15% over Q1",
        "confidence": 0.96,
        "timestamp": "2025-04-03"
    },
    "quarterly_trend": {
        "value": "Last quarter shows continued growth from previous quarters",
        "confidence": 0.98,
        "timestamp": "2025-04-03"
    },
    "comparative_metrics": {
        "value": "Q1: 2.1M, Q2: 2.4M, representing 3 quarters of growth",
        "confidence": 0.95,
        "timestamp": "2025-04-03"
    },
    "time_period": {
        "value": "Last quarter (Q2) compared with previous quarters shows strong growth",
        "confidence": 0.99,
        "timestamp": "2025-04-03"
    }
}

# Fused data specifically for intent fusion (benchmark case 06)
fused_business_data = {
    # Sales summary data
    "q2_revenue": {
        "value": "$2.4M in Q2 revenue, up 12% from Q1",
        "confidence": 0.97,
        "timestamp": "2025-04-01"
    },
    "top_products": {
        "value": "Premium subscription showing 22% growth versus last quarter",
        "confidence": 0.96,
        "timestamp": "2025-04-01"
    },
    
    # Comparative analysis data
    "q1_to_q2_growth": {
        "value": "Q2 showed 12% revenue growth compared to Q1, with consistent monthly increases",
        "confidence": 0.98,
        "timestamp": "2025-04-03"
    },
    "growth_drivers": {
        "value": "Primary growth drivers: new customer acquisition (+7%) and premium tier upgrades (+22%)",
        "confidence": 0.95,
        "timestamp": "2025-04-03"
    },
    
    # Context for both intents
    "time_period": {
        "value": "Last quarter (Q2) data compared with Q1 shows accelerating growth trends",
        "confidence": 0.99,
        "timestamp": "2025-04-03"
    },
    "market_context": {
        "value": "Growth occurred despite 3% market contraction in Q2, indicating increased market share",
        "confidence": 0.94,
        "timestamp": "2025-04-02"
    },
    
    # Business insights connecting both summary and comparison
    "business_outlook": {
        "value": "Current growth trajectory suggests Q3 will exceed projections by 5-8% if current trends continue",
        "confidence": 0.91,
        "timestamp": "2025-04-04"
    }
}

# Risk analysis data for benchmark case 07
risk_analysis_data = {
    # Operational risks
    "operational_risks": {
        "value": "Supply chain disruptions affecting 3 key vendors, with 15% increased lead times",
        "confidence": 0.93,
        "timestamp": "2025-04-10"
    },
    "staffing_vulnerabilities": {
        "value": "Critical team turnover at 8%, up from 5% last quarter, concentrated in engineering",
        "confidence": 0.91,
        "timestamp": "2025-04-09"
    },
    "process_failures": {
        "value": "Quality control incident rate increased to 1.2%, exceeding target threshold of 0.8%",
        "confidence": 0.94,
        "timestamp": "2025-04-08"
    },
    
    # Financial risks
    "revenue_volatility": {
        "value": "Revenue variance of 12% against forecast, primarily in APAC region",
        "confidence": 0.88,
        "timestamp": "2025-04-11"
    },
    "margin_pressure": {
        "value": "Gross margin declined 2.3 percentage points due to input cost inflation",
        "confidence": 0.92,
        "timestamp": "2025-04-10"
    },
    "currency_exposure": {
        "value": "FX headwinds resulted in 3.5% negative impact on international revenues",
        "confidence": 0.89,
        "timestamp": "2025-04-09"
    },
    
    # Strategic risks
    "competitive_threats": {
        "value": "New market entrant captured 5% share in enterprise segment within 90 days",
        "confidence": 0.87,
        "timestamp": "2025-04-11"
    },
    "product_lifecycle": {
        "value": "Core product line showing signs of maturity with growth rate declined to 4% from 12%",
        "confidence": 0.90,
        "timestamp": "2025-04-10"
    },
    "regulatory_changes": {
        "value": "Upcoming data privacy regulations will require compliance overhaul by Q3",
        "confidence": 0.95,
        "timestamp": "2025-04-08"
    },
    
    # Time context
    "time_period": {
        "value": "Q2 2025 risk assessment identifies 3 critical, 4 significant, and 5 moderate risks",
        "confidence": 0.97,
        "timestamp": "2025-04-12"
    },
    
    # Risk metrics
    "risk_metrics": {
        "value": "Composite risk score of 72/100, up from 65/100 in Q1, exceeding risk appetite threshold",
        "confidence": 0.96,
        "timestamp": "2025-04-12"
    }
}

# Performance data for lifecycle benchmark (case 08)
# Session 1 (Q1) - Focused on immediate issues
lifecycle_performance_q1 = {
    "current_issues": {
        "value": "Three departments showing 15% below target KPIs this quarter",
        "confidence": 0.92,
        "timestamp": "2025-01-15",
        "age": 1
    },
    "immediate_concerns": {
        "value": "Product launch delays affecting Q1 revenue projections by 8%",
        "confidence": 0.88,
        "timestamp": "2025-01-14",
        "age": 1
    },
    "team_performance": {
        "value": "Engineering team velocity down 12% from previous quarter",
        "confidence": 0.90,
        "timestamp": "2025-01-15",
        "age": 1
    },
    "resource_allocation": {
        "value": "Budget overruns in marketing (18%) and R&D (7%) departments",
        "confidence": 0.93,
        "timestamp": "2025-01-13",
        "age": 1
    },
    "critical_metrics": {
        "value": "Customer satisfaction score dipped to 82%, below 85% threshold",
        "confidence": 0.89,
        "timestamp": "2025-01-15",
        "age": 1
    },
    "time_period": {
        "value": "Current quarter (Q1) showing several performance issues requiring immediate attention",
        "confidence": 0.95,
        "timestamp": "2025-01-15",
        "age": 1
    }
}

# Session 2 (Q2) - Beginning to look for patterns
lifecycle_performance_q2 = {
    "performance_trends": {
        "value": "Q1 to Q2 trend analysis shows 7% improvement in overall KPI achievement",
        "confidence": 0.91,
        "timestamp": "2025-04-15",
        "age": 1
    },
    "recurring_patterns": {
        "value": "Sales team consistently underperforms in first month of each quarter by ~12%",
        "confidence": 0.87,
        "timestamp": "2025-04-14",
        "age": 1
    },
    "cross_department_analysis": {
        "value": "Marketing spend efficiency correlates strongly (0.82) with sales performance",
        "confidence": 0.89,
        "timestamp": "2025-04-15",
        "age": 1
    },
    "historical_comparison": {
        "value": "Current patterns show similarity to company performance during product transition phases",
        "confidence": 0.85,
        "timestamp": "2025-04-13",
        "age": 1
    },
    "productivity_indicators": {
        "value": "Developer productivity increased 15% after addressing technical debt identified in Q1",
        "confidence": 0.92,
        "timestamp": "2025-04-15",
        "age": 1
    },
    "emergent_issues": {
        "value": "New competitor products beginning to impact market share in 2 of 5 segments",
        "confidence": 0.88,
        "timestamp": "2025-04-14",
        "age": 1
    },
    "time_period": {
        "value": "Analysis comparing Q1 and Q2 performance metrics across departments",
        "confidence": 0.96,
        "timestamp": "2025-04-15",
        "age": 1
    }
}

# Session 3 (Q3) - Forward-looking with strategic intent
lifecycle_performance_q3 = {
    "strategic_indicators": {
        "value": "Three emerging market opportunities identified with combined TAM of $2.7B",
        "confidence": 0.93,
        "timestamp": "2025-07-20",
        "age": 1
    },
    "performance_trajectory": {
        "value": "Current growth trajectory projects 22% increase in annual revenue if Q1-Q3 trends continue",
        "confidence": 0.89,
        "timestamp": "2025-07-19",
        "age": 1
    },
    "market_conditions": {
        "value": "Industry analysis indicates favorable market conditions for next 18 months",
        "confidence": 0.86,
        "timestamp": "2025-07-18",
        "age": 1
    },
    "competitive_landscape": {
        "value": "Company positioned as innovation leader in 2 of 4 core markets based on Q3 analysis",
        "confidence": 0.91,
        "timestamp": "2025-07-20",
        "age": 1
    },
    "resource_planning": {
        "value": "Strategic resource allocation model recommends 15% increased investment in AI capabilities",
        "confidence": 0.94,
        "timestamp": "2025-07-17",
        "age": 1
    },
    "risk_assessment": {
        "value": "Forward-looking risk analysis identifies 3 high-impact, low-probability strategic threats",
        "confidence": 0.90,
        "timestamp": "2025-07-19",
        "age": 1
    },
    "long_term_outlook": {
        "value": "Five-year projection based on current trajectory shows 3.5x growth potential",
        "confidence": 0.85,
        "timestamp": "2025-07-20",
        "age": 1
    },
    "time_period": {
        "value": "Forward-looking strategic analysis covering next 18-month horizon based on Q1-Q3 trends",
        "confidence": 0.95,
        "timestamp": "2025-07-20",
        "age": 1
    }
}
