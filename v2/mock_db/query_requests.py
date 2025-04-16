query_requests = [
    {
        "query": "Compile performance insights across departments and align them with team engagement trends.",
        "timestamp": "2025-04-12T09:15:00Z",
        "user_id": "user_001",
        "department": "strategy"
    },
    {
        "query": "Compare Q2 output from engineering with sales targets.",
        "timestamp": "2025-04-12T09:20:00Z",
        "user_id": "user_002",
        "department": "product"
    },
    {
        "query": "Evaluate team sentiment fluctuations over the last quarter.",
        "timestamp": "2025-04-12T09:25:00Z",
        "user_id": "user_003",
        "department": "hr"
    },
    {
        "query": "Highlight anomalies in productivity across business units.",
        "timestamp": "2025-04-12T09:30:00Z",
        "user_id": "user_004",
        "department": "ops"
    }
]

# Multi-intent query that combines summarizing sales data and tracking inventory
multi_intent_query_request = {
    "query": "Summarize Q2 sales performance and analyze inventory trends. Compare both to previous quarters.",
    "timestamp": "2025-04-12T10:15:00Z",
    "user_id": "user_001",
    "department": "executive",
    "priority": "high",
    "context": "quarterly_review"
}

# Fused intent query for benchmark case 06
fused_query_request = {
    "query": "Summarize Q2 sales and compare them with Q1 to identify growth trends.",
    "timestamp": "2025-04-12T11:30:00Z",
    "user_id": "user_002",
    "department": "business_intelligence",
    "priority": "medium",
    "context": "growth_analysis",
    "decision_style": "comparative"
}

# Memory-conditioned query for benchmark case 07
memory_conditioned_query = {
    "query": "Given my past analysis style, summarize this quarter's risks.",
    "timestamp": "2025-04-12T14:45:00Z",
    "user_id": "risk_analyst_001",
    "department": "enterprise_risk",
    "priority": "medium",
    "context": "quarterly_risk_review",
    "decision_style": "assertive",
    "memory_trace": {
        "session_history": [
            {
                "session_id": "risk_session_001",
                "timestamp": "2025-01-15T10:30:00Z",
                "risk_focus": "operational",
                "decision_style": "assertive",
                "confidence_threshold_used": 0.75,
                "semantic_class": "RiskProfile"
            },
            {
                "session_id": "risk_session_002",
                "timestamp": "2025-02-18T09:45:00Z",
                "risk_focus": "financial",
                "decision_style": "cautious",
                "confidence_threshold_used": 0.82,
                "semantic_class": "RiskProfile"
            },
            {
                "session_id": "risk_session_003",
                "timestamp": "2025-03-20T11:15:00Z",
                "risk_focus": "strategic",
                "decision_style": "balanced",
                "confidence_threshold_used": 0.78,
                "semantic_class": "RiskProfile" 
            }
        ],
        "preference_history": {
            "always_include_metrics": True,
            "highlight_critical_risks": True,
            "prefer_comparative_analysis": True,
            "detail_level": "high"
        },
        "learned_traits": {
            "assertiveness": 0.85,
            "risk_tolerance": 0.35,
            "analytical_depth": 0.92,
            "evidence_threshold": 0.80,
            "follow_up_frequency": 0.75
        }
    }
}

# Lifecycle queries for benchmark case 08 - showing user evolution over time
lifecycle_queries = [
    # Session 1: Reactive phase - Focused on immediate issues
    {
        "query": "What are our current performance issues this quarter?",
        "timestamp": "2025-01-15T10:30:00Z",
        "user_id": "strategic_planner_001",
        "department": "strategic_planning",
        "priority": "medium",
        "context": "quarterly_review",
        "decision_style": "reactive",
        "session_number": 1,
        "quarter": "Q1",
        "memory_trace": {
            "session_history": [],  # Empty in first session
            "preference_history": {
                "detail_level": "low",
                "prefer_tactical_solutions": True,
                "focus_on_problems": True,
                "require_immediate_actions": True
            },
            "learned_traits": {
                "analytical_depth": 0.45,
                "strategic_orientation": 0.30,
                "risk_tolerance": 0.60,
                "decisiveness": 0.75,
                "planning_horizon": 0.25  # Short-term focus
            }
        }
    },
    
    # Session 2: Transition phase - Beginning to look for patterns
    {
        "query": "Analyze performance trends across departments and identify recurring patterns from last quarter.",
        "timestamp": "2025-04-15T11:15:00Z",
        "user_id": "strategic_planner_001",
        "department": "strategic_planning",
        "priority": "high",
        "context": "strategic_planning",
        "decision_style": "analytical",
        "session_number": 2,
        "quarter": "Q2",
        "memory_trace": {
            "session_history": [
                {
                    "session_id": "planning_session_001",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "focus": "tactical",
                    "decision_style": "reactive",
                    "outcomes": "Identified immediate issues but missed underlying patterns",
                    "confidence_threshold_used": 0.65,
                    "semantic_class": "PerformanceAnalysis"
                }
            ],
            "preference_history": {
                "detail_level": "medium",
                "prefer_tactical_solutions": False,
                "focus_on_problems": True,
                "require_immediate_actions": False,
                "seeking_patterns": True
            },
            "learned_traits": {
                "analytical_depth": 0.65,
                "strategic_orientation": 0.55,
                "risk_tolerance": 0.55,
                "decisiveness": 0.70,
                "planning_horizon": 0.60  # Medium-term focus
            }
        }
    },
    
    # Session 3: Proactive phase - Forward-looking with strategic intent
    {
        "query": "Develop a forward-looking strategic plan based on historical performance trends and projected market conditions.",
        "timestamp": "2025-07-20T09:45:00Z",
        "user_id": "strategic_planner_001",
        "department": "strategic_planning",
        "priority": "critical",
        "context": "annual_planning",
        "decision_style": "strategic",
        "session_number": 3,
        "quarter": "Q3",
        "memory_trace": {
            "session_history": [
                {
                    "session_id": "planning_session_001",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "focus": "tactical",
                    "decision_style": "reactive",
                    "outcomes": "Identified immediate issues but missed underlying patterns",
                    "confidence_threshold_used": 0.65,
                    "semantic_class": "PerformanceAnalysis"
                },
                {
                    "session_id": "planning_session_002",
                    "timestamp": "2025-04-15T11:15:00Z",
                    "focus": "analytical",
                    "decision_style": "analytical",
                    "outcomes": "Discovered recurring performance patterns across departments",
                    "confidence_threshold_used": 0.75,
                    "semantic_class": "PerformanceAnalysis" 
                }
            ],
            "preference_history": {
                "detail_level": "high",
                "prefer_tactical_solutions": False,
                "focus_on_problems": False,
                "require_immediate_actions": False,
                "seeking_patterns": True,
                "forward_looking": True,
                "strategic_orientation": True
            },
            "learned_traits": {
                "analytical_depth": 0.85,
                "strategic_orientation": 0.90,
                "risk_tolerance": 0.50,
                "decisiveness": 0.80,
                "planning_horizon": 0.85  # Long-term focus
            }
        }
    }
]