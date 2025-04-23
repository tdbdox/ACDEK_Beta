"""
Domain Configurations for ACDEK v3

This module provides standardized domain classifications and configurations
for use across the entire system. This ensures consistent domain handling
in benchmarks, grounding, constraint verification, and other components.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Set


class Domain(Enum):
    """
    Standard domain classification for all ACDEK v3 components.
    
    All components should import and use this enum rather than
    defining their own Domain enums to ensure consistency.
    """
    MEDICAL = "medical"         # Medical and healthcare domain
    LEGAL = "legal"             # Legal and regulatory domain
    FINANCIAL = "financial"     # Financial and economic domain
    SCIENTIFIC = "scientific"   # Scientific research domain
    TECHNICAL = "technical"     # Technical and engineering domain
    HISTORICAL = "historical"   # Historical information domain
    ACADEMIC = "academic"       # Academic and educational domain
    GENERAL = "general"         # General purpose domain
    
    @classmethod
    def from_string(cls, domain_str: str) -> 'Domain':
        """
        Convert a string representation to a Domain enum.
        
        Args:
            domain_str: String representation of the domain
            
        Returns:
            Domain enum value
            
        Raises:
            ValueError if the string doesn't match any domain
        """
        try:
            return cls(domain_str.lower())
        except ValueError:
            # Handle common variations or aliases
            domain_map = {
                "medicine": cls.MEDICAL,
                "healthcare": cls.MEDICAL,
                "health": cls.MEDICAL,
                "law": cls.LEGAL,
                "regulatory": cls.LEGAL,
                "finance": cls.FINANCIAL,
                "economic": cls.FINANCIAL,
                "economics": cls.FINANCIAL,
                "science": cls.SCIENTIFIC,
                "research": cls.SCIENTIFIC,
                "tech": cls.TECHNICAL,
                "engineering": cls.TECHNICAL,
                "software": cls.TECHNICAL,
                "history": cls.HISTORICAL,
                "education": cls.ACADEMIC,
                "educational": cls.ACADEMIC
            }
            
            normalized = domain_str.lower()
            if normalized in domain_map:
                return domain_map[normalized]
            
            # Default to general if no match found
            return cls.GENERAL
    
    @classmethod
    def get_all_values(cls) -> List[str]:
        """Get all domain values as strings."""
        return [domain.value for domain in cls]
    
    @classmethod
    def get_domain_info(cls, domain: 'Domain') -> Dict[str, Any]:
        """
        Get additional information about a specific domain.
        
        Args:
            domain: The domain to get info for
            
        Returns:
            Dictionary with domain metadata
        """
        domain_info = {
            cls.MEDICAL: {
                "description": "Medical and healthcare information",
                "requires_citations": True,
                "sensitive": True,
                "confidence_threshold": 0.85,
                "fallback_threshold": 0.75
            },
            cls.LEGAL: {
                "description": "Legal and regulatory information",
                "requires_citations": True,
                "sensitive": True,
                "confidence_threshold": 0.85,
                "fallback_threshold": 0.75
            },
            cls.FINANCIAL: {
                "description": "Financial and economic information",
                "requires_citations": True,
                "sensitive": True,
                "confidence_threshold": 0.80,
                "fallback_threshold": 0.70
            },
            cls.SCIENTIFIC: {
                "description": "Scientific research information",
                "requires_citations": True,
                "sensitive": False,
                "confidence_threshold": 0.80,
                "fallback_threshold": 0.70
            },
            cls.TECHNICAL: {
                "description": "Technical and engineering information",
                "requires_citations": False,
                "sensitive": False,
                "confidence_threshold": 0.75,
                "fallback_threshold": 0.65
            },
            cls.HISTORICAL: {
                "description": "Historical information",
                "requires_citations": True,
                "sensitive": False,
                "confidence_threshold": 0.75,
                "fallback_threshold": 0.65
            },
            cls.ACADEMIC: {
                "description": "Academic and educational information",
                "requires_citations": True,
                "sensitive": False,
                "confidence_threshold": 0.75,
                "fallback_threshold": 0.65
            },
            cls.GENERAL: {
                "description": "General purpose information",
                "requires_citations": False,
                "sensitive": False,
                "confidence_threshold": 0.70,
                "fallback_threshold": 0.60
            }
        }
        
        return domain_info.get(domain, domain_info[cls.GENERAL])


# Default threshold configurations by domain
DEFAULT_THRESHOLDS = {
    Domain.MEDICAL: {
        "factuality": 0.85,
        "hallucination": 0.85,
        "confidence": 0.80,
        "constraint": 0.90,
        "latency_ms": 500
    },
    Domain.LEGAL: {
        "factuality": 0.85,
        "hallucination": 0.85,
        "confidence": 0.80,
        "constraint": 0.90,
        "latency_ms": 500
    },
    Domain.FINANCIAL: {
        "factuality": 0.80,
        "hallucination": 0.85,
        "confidence": 0.75,
        "constraint": 0.85,
        "latency_ms": 450
    },
    Domain.SCIENTIFIC: {
        "factuality": 0.80,
        "hallucination": 0.80,
        "confidence": 0.75,
        "constraint": 0.85,
        "latency_ms": 500
    },
    Domain.TECHNICAL: {
        "factuality": 0.75,
        "hallucination": 0.75, 
        "confidence": 0.70,
        "constraint": 0.80,
        "latency_ms": 400
    },
    Domain.HISTORICAL: {
        "factuality": 0.80,
        "hallucination": 0.80,
        "confidence": 0.75,
        "constraint": 0.85,
        "latency_ms": 450
    },
    Domain.ACADEMIC: {
        "factuality": 0.80,
        "hallucination": 0.80,
        "confidence": 0.75,
        "constraint": 0.85,
        "latency_ms": 450
    },
    Domain.GENERAL: {
        "factuality": 0.70,
        "hallucination": 0.70,
        "confidence": 0.65,
        "constraint": 0.75,
        "latency_ms": 350
    }
}


def get_threshold(domain: Domain, threshold_type: str) -> float:
    """
    Get a specific threshold for a domain.
    
    Args:
        domain: The domain to get threshold for
        threshold_type: Type of threshold (factuality, hallucination, etc.)
        
    Returns:
        Threshold value as float
    """
    domain_thresholds = DEFAULT_THRESHOLDS.get(domain, DEFAULT_THRESHOLDS[Domain.GENERAL])
    return domain_thresholds.get(threshold_type, 0.7)  # Default to 0.7 if not found


def get_domain_thresholds(domain: Domain) -> Dict[str, float]:
    """
    Get all thresholds for a specific domain.
    
    Args:
        domain: The domain to get thresholds for
        
    Returns:
        Dictionary of threshold values
    """
    return DEFAULT_THRESHOLDS.get(domain, DEFAULT_THRESHOLDS[Domain.GENERAL])