#!/usr/bin/env python3
"""
Real-World Dataset Loader for ACDEK v3 Grounding Tests

This module provides utilities for loading and using real-world test datasets
for evaluating the grounding engine's hallucination detection capabilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_world_data_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealWorldDataLoader:
    """Loader for real-world test datasets for grounding evaluation."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the real-world data loader.
        
        Args:
            base_path: Optional override for the base data path
        """
        # Determine base directory for data
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Try to find data directory relative to this file
            script_dir = Path(__file__).resolve().parent
            parent_dir = script_dir.parent
            
            # Default to v3-data/data/real_world_test
            self.base_path = parent_dir / "data" / "real_world_test"
        
        logger.info(f"Initialized data loader with data directory: {self.base_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        
        # Find available domains
        self._available_domains = self._discover_domains()
    
    @property
    def available_domains(self) -> List[str]:
        """Get list of available domains with test data"""
        return self._available_domains
    
    def _discover_domains(self) -> List[str]:
        """
        Discover domains with available test data
        
        Returns:
            List of domain names
        """
        domains = []
        
        # Look for JSON files in the data directory
        try:
            for file_path in self.base_path.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    domain = data.get("domain")
                    if domain:
                        domains.append(domain)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error discovering domains: {e}")
        
        logger.info(f"Discovered domains: {domains}")
        return domains
    
    def load_test_cases(self, domain: str) -> List[Dict[str, Any]]:
        """
        Load test cases for a specific domain
        
        Args:
            domain: Domain name to load test cases for
            
        Returns:
            List of test cases
        """
        test_cases = []
        
        # Find the JSON file for the domain
        domain_file = self.base_path / f"{domain}_test.json"
        
        if not domain_file.exists():
            logger.warning(f"No test data file found for domain '{domain}'")
            return []
        
        try:
            with open(domain_file, 'r') as f:
                data = json.load(f)
            
            # Extract test cases
            if data.get("domain") == domain:
                test_cases.extend(data.get("test_cases", []))
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading test cases for domain '{domain}': {e}")
        
        logger.info(f"Loaded {len(test_cases)} test cases for domain '{domain}'")
        return test_cases
    
    def create_example_datasets(self, overwrite: bool = False):
        """
        Create example test datasets if no real-world data is available
        
        Args:
            overwrite: Whether to overwrite existing files
        """
        # Create test datasets for medical, financial, and legal domains
        domains = ["medical", "financial", "legal"]
        examples_created = 0
        
        for domain in domains:
            output_file = self.base_path / f"{domain}_test.json"
            
            # Skip if file already exists and not overwriting
            if output_file.exists() and not overwrite:
                logger.info(f"Example dataset for {domain} already exists at {output_file}")
                continue
            
            # Generate example test cases
            example_data = self._generate_example_test_cases(domain)
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(example_data, f, indent=2)
                
                logger.info(f"Created example dataset for {domain} at {output_file}")
                examples_created += 1
            except IOError as e:
                logger.error(f"Error creating example dataset for {domain}: {e}")
        
        logger.info(f"Created {examples_created} example datasets")
        
        # Refresh available domains
        self._available_domains = self._discover_domains()
    
    def _generate_example_test_cases(self, domain: str) -> Dict[str, Any]:
        """
        Generate example test cases for a domain
        
        Args:
            domain: Domain to generate test cases for
            
        Returns:
            Dictionary with domain and test cases
        """
        if domain == "medical":
            return {
                "domain": "medical",
                "test_cases": [
                    {
                        "id": "med-001",
                        "prompt": "What are the symptoms of type 2 diabetes?",
                        "reference_text": "Type 2 diabetes symptoms include increased thirst, frequent urination, hunger, fatigue, blurred vision, slow-healing sores, frequent infections, and areas of darkened skin. It develops when the body becomes resistant to insulin or when the pancreas doesn't produce enough insulin.",
                        "expert_judgment": "factual",
                        "explanation": "This is a standard medical description of type 2 diabetes symptoms."
                    },
                    {
                        "id": "med-002",
                        "prompt": "Does vitamin C cure cancer?",
                        "reference_text": "There is no strong scientific evidence that vitamin C alone can cure cancer. While some studies suggest vitamin C may enhance the effects of certain cancer treatments or have some anti-cancer properties in specific contexts, major cancer organizations do not recognize vitamin C as a standalone cancer cure.",
                        "expert_judgment": "hallucination",
                        "explanation": "Claims that vitamin C cures cancer are not supported by scientific consensus."
                    },
                    {
                        "id": "med-003",
                        "prompt": "How do vaccines work?",
                        "reference_text": "Vaccines work by stimulating the immune system to recognize and fight specific disease-causing organisms. They typically contain weakened or inactivated forms of the pathogen, or parts of it like proteins or sugars. When introduced to the body, these substances trigger an immune response without causing the disease. The immune system produces antibodies and memory cells that remember how to fight that specific pathogen if exposed to it in the future.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of how vaccines work."
                    },
                    {
                        "id": "med-004",
                        "prompt": "Do vaccines cause autism?",
                        "reference_text": "Extensive scientific research has found no link between vaccines and autism. Multiple large-scale studies have investigated potential connections between vaccines and autism spectrum disorders and have consistently found no association. The original study that suggested a link has been retracted due to serious procedural and ethical flaws, and its author lost his medical license.",
                        "expert_judgment": "hallucination",
                        "explanation": "Claims linking vaccines to autism contradict extensive scientific evidence showing no connection."
                    },
                    {
                        "id": "med-005",
                        "prompt": "What is the function of the heart?",
                        "reference_text": "The heart is a muscular organ that pumps blood throughout the body via the circulatory system. It delivers oxygen and nutrients to tissues and removes carbon dioxide and other wastes. The heart has four chambers: two atria (upper) and two ventricles (lower). The right side of the heart receives deoxygenated blood from the body and pumps it to the lungs, while the left side receives oxygenated blood from the lungs and pumps it to the rest of the body.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of the heart's function."
                    }
                ]
            }
        elif domain == "financial":
            return {
                "domain": "financial",
                "test_cases": [
                    {
                        "id": "fin-001",
                        "prompt": "What is compound interest?",
                        "reference_text": "Compound interest is the interest on a loan or deposit calculated based on both the initial principal and the accumulated interest from previous periods. Unlike simple interest, which is calculated only on the principal amount, compound interest generates interest on interest, leading to exponential growth over time. The formula for compound interest is A = P(1 + r/n)^(nt), where A is the final amount, P is the principal, r is the interest rate, n is the number of times interest is compounded per period, and t is the number of periods.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of compound interest and its formula."
                    },
                    {
                        "id": "fin-002",
                        "prompt": "The FDIC insures all investments up to $1 million per account",
                        "reference_text": "The Federal Deposit Insurance Corporation (FDIC) insures deposits in member banks up to $250,000 per depositor, per insured bank, for each account ownership category. The FDIC does not insure securities, mutual funds, or similar types of investments that banks and thrift institutions may offer, such as stocks, bonds, money market funds, life insurance policies, annuities, or municipal securities.",
                        "expert_judgment": "hallucination",
                        "explanation": "This statement contains multiple inaccuracies about FDIC coverage limits and what is covered."
                    },
                    {
                        "id": "fin-003",
                        "prompt": "What is a bond in finance?",
                        "reference_text": "A bond is a fixed income instrument representing a loan made by an investor to a borrower, typically corporate or governmental entities. When you buy a bond, you are lending money to the issuer for a specified period at a fixed interest rate. The bond issuer promises to pay periodic interest payments (known as coupon payments) and to repay the principal on a specified date (maturity date). Bonds are used by companies, municipalities, states, and governments to finance projects and operations.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of bonds in finance."
                    },
                    {
                        "id": "fin-004",
                        "prompt": "Bitcoin was created by the Federal Reserve in 2010",
                        "reference_text": "Bitcoin was created in 2009 by an unknown person or group of people using the pseudonym Satoshi Nakamoto. It was introduced as open-source software and operates as a decentralized digital currency without a central bank or administrator. Bitcoin was not created by the Federal Reserve or any government entity, and its design is specifically intended to operate independently of central banking systems.",
                        "expert_judgment": "hallucination",
                        "explanation": "This statement contains multiple factual errors about Bitcoin's creation and purpose."
                    },
                    {
                        "id": "fin-005",
                        "prompt": "What is diversification in investing?",
                        "reference_text": "Diversification is an investment strategy that involves spreading investments across various financial instruments, industries, and categories to reduce risk. The principle is that a portfolio constructed of different kinds of assets will, on average, yield higher long-term returns and lower the risk of any individual holding or security. Diversification seeks to reduce unsystematic risk events in a portfolio so that the positive performance of some investments neutralizes the negative performance of others.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of diversification in investing."
                    }
                ]
            }
        elif domain == "legal":
            return {
                "domain": "legal",
                "test_cases": [
                    {
                        "id": "leg-001",
                        "prompt": "What is habeas corpus?",
                        "reference_text": "Habeas corpus is a legal procedure that allows individuals to challenge their detention or imprisonment before a court. The term comes from Latin and roughly translates to 'you shall have the body.' It is a fundamental legal remedy that protects against unlawful detention by requiring that a prisoner be brought before a court to determine if the person's imprisonment or detention is lawful. In the United States, the right to habeas corpus is enshrined in the Constitution and can only be suspended in cases of rebellion or invasion.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of habeas corpus."
                    },
                    {
                        "id": "leg-002",
                        "prompt": "Every state in the US has the death penalty",
                        "reference_text": "Not every state in the United States has the death penalty. As of 2023, capital punishment has been abolished in 23 states, while it remains legal in 27 states. Additionally, the federal government and the U.S. military maintain the death penalty. States that have abolished the death penalty include Alaska, Colorado, Connecticut, Delaware, Hawaii, Illinois, Iowa, Maine, Maryland, Massachusetts, Michigan, Minnesota, New Hampshire, New Jersey, New Mexico, New York, North Dakota, Rhode Island, Vermont, Virginia, Washington, West Virginia, and Wisconsin.",
                        "expert_judgment": "hallucination",
                        "explanation": "This statement is factually incorrect as many states have abolished the death penalty."
                    },
                    {
                        "id": "leg-003",
                        "prompt": "What is the difference between a felony and a misdemeanor?",
                        "reference_text": "In the United States legal system, crimes are generally classified as either felonies or misdemeanors based on their severity. Felonies are more serious crimes that are typically punishable by imprisonment of more than one year in a state or federal prison, and may also include substantial fines. Examples include murder, rape, burglary, and fraud above certain monetary thresholds. Misdemeanors are less serious offenses that are generally punishable by up to one year in county jail, smaller fines, probation, or community service. Examples include petty theft, simple assault, or driving under the influence (in many jurisdictions for first offenses).",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of the distinction between felonies and misdemeanors."
                    },
                    {
                        "id": "leg-004",
                        "prompt": "In the US, defendants must prove their innocence in criminal trials",
                        "reference_text": "In the United States criminal justice system, defendants are presumed innocent until proven guilty. The burden of proof lies with the prosecution, which must prove the defendant's guilt 'beyond a reasonable doubt.' Defendants do not have to prove their innocence, and they have the right to remain silent without this being used against them, as protected by the Fifth Amendment. This foundational principle is designed to protect individuals from wrongful conviction and is a cornerstone of the American criminal justice system.",
                        "expert_judgment": "hallucination",
                        "explanation": "This statement directly contradicts the presumption of innocence principle in US criminal law."
                    },
                    {
                        "id": "leg-005",
                        "prompt": "What is tort law?",
                        "reference_text": "Tort law is the branch of law that deals with civil wrongs committed by one party against another, outside of contractual obligations. It provides remedies, usually in the form of monetary compensation, to those who have suffered harm due to the wrongful acts of others. Common types of torts include negligence (failing to exercise reasonable care), intentional torts (such as assault, battery, or defamation), and strict liability torts (where fault is not required, such as in some product liability cases). The primary aim of tort law is to provide relief for the damages incurred and to deter others from committing similar harms.",
                        "expert_judgment": "factual",
                        "explanation": "This is an accurate description of tort law."
                    }
                ]
            }
        else:
            return {
                "domain": domain,
                "test_cases": [
                    {
                        "id": f"{domain}-001",
                        "prompt": f"Example prompt for {domain}",
                        "reference_text": f"Example reference text for {domain}",
                        "expert_judgment": "factual",
                        "explanation": "This is an example test case."
                    }
                ]
            }

def create_example_datasets():
    """Create example test datasets if run directly"""
    data_loader = RealWorldDataLoader()
    data_loader.create_example_datasets()

if __name__ == "__main__":
    create_example_datasets()