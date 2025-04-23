"""
Domain-specific test cases for ACDEK v3 benchmarking

This module provides specialized test case sets for different domains:
1. Medical
2. Legal
3. Financial
4. Technical
5. Edge cases for benchmarking

Each test case includes ground truth statements, reference content, constraints,
and expected hallucination patterns for comprehensive benchmarking.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class Domain(Enum):
    """Domains for benchmark test cases."""
    MEDICAL = "medical"
    LEGAL = "legal" 
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    GENERAL = "general"
    EDGE_CASE = "edge_case"


class HallucinationType(Enum):
    """Categories of hallucinations for detecting specific issues."""
    FACTUAL_ERROR = "factual_error"         # Incorrect facts
    FACTUAL_FABRICATION = "fabrication"     # Made-up facts
    LOGICAL_INCONSISTENCY = "logical"       # Logical errors or contradictions
    TEMPORAL_ERROR = "temporal"             # Time-related errors
    NUMERICAL_ERROR = "numerical"           # Incorrect numbers or statistics
    ENTITY_ERROR = "entity"                 # Incorrect entities, people, places
    CONTEXT_DRIFT = "context_drift"         # Drifting from original context
    CONSTRAINT_VIOLATION = "constraint"     # Violates specific constraints
    CONFIDENCE_MISMATCH = "confidence"      # Confidence doesn't match factuality


@dataclass
class ReferenceContent:
    """Reference content for factuality checking."""
    content: str
    source: str
    source_type: str = "text"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type
        }


@dataclass
class Constraint:
    """Constraint for validation."""
    constraint_type: str
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.constraint_type,
            "required": self.required,
            **self.metadata
        }


@dataclass
class GroundTruth:
    """Ground truth data for test cases."""
    facts: List[str]
    constraints: List[Constraint]
    invalid_claims: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "facts": self.facts,
            "constraints": [c.to_dict() for c in self.constraints],
            "invalid_claims": self.invalid_claims,
            "citations": self.citations
        }


@dataclass
class TestCase:
    """Test case definition for benchmarking."""
    id: str
    prompt: str
    domain: Domain
    ground_truth: GroundTruth
    reference_content: List[ReferenceContent]
    difficulty: int = 1  # 1-5 scale (5 being most difficult)
    tags: List[str] = field(default_factory=list)
    expected_hallucination_types: List[HallucinationType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "domain": self.domain.value,
            "ground_truth": self.ground_truth.to_dict(),
            "reference_content": [r.to_dict() for r in self.reference_content],
            "difficulty": self.difficulty,
            "tags": self.tags,
            "expected_hallucination_types": [h.value for h in self.expected_hallucination_types]
        }


# Medical domain test cases
def get_medical_test_cases() -> List[TestCase]:
    """Get medical domain test cases."""
    return [
        # Basic medical case - diabetes management
        TestCase(
            id="med-001",
            prompt="What are the management options for Type 2 diabetes?",
            domain=Domain.MEDICAL,
            ground_truth=GroundTruth(
                facts=[
                    "Type 2 diabetes management includes lifestyle modifications like diet and exercise.",
                    "Medications such as metformin are commonly used as first-line treatment.",
                    "Regular blood glucose monitoring is essential for diabetes management.",
                    "Insulin may be required in some cases of Type 2 diabetes."
                ],
                constraints=[
                    Constraint(constraint_type="source_citation", required=True),
                    Constraint(constraint_type="avoid_guarantees", required=True)
                ],
                invalid_claims=[
                    "Type 2 diabetes can always be cured",
                    "Type 2 diabetes only requires medication and no lifestyle changes",
                    "All patients with Type 2 diabetes will eventually need insulin"
                ],
                citations=[
                    "American Diabetes Association",
                    "Mayo Clinic",
                    "WHO Guidelines"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Type 2 diabetes management focuses on lifestyle modifications including healthy diet, regular physical activity, and maintaining a healthy weight. These lifestyle changes may be enough to control blood sugar levels for some people. If not, medications such as metformin are typically prescribed as first-line therapy. Regular monitoring of blood glucose levels is essential. In some cases, insulin therapy or other injectable medications may be required.",
                    source="American Diabetes Association Guidelines 2023"
                ),
                ReferenceContent(
                    content="The primary goal of Type 2 diabetes management is to keep blood sugar levels within a healthy range. This typically involves lifestyle modifications (diet, exercise, weight management), oral medications like metformin, and sometimes insulin. Regular monitoring of blood glucose and regular medical check-ups are essential components of diabetes care.",
                    source="Mayo Clinic Diabetes Management Guide"
                )
            ],
            difficulty=1,
            tags=["diabetes", "chronic disease", "medication"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.CONFIDENCE_MISMATCH
            ]
        ),
        
        # Complex medical case - cancer treatment
        TestCase(
            id="med-002",
            prompt="What are the latest advances in breast cancer treatment? Include information about targeted therapies.",
            domain=Domain.MEDICAL,
            ground_truth=GroundTruth(
                facts=[
                    "HER2-targeted therapies like trastuzumab improve outcomes for HER2-positive breast cancer.",
                    "CDK4/6 inhibitors have shown benefit in hormone receptor-positive metastatic breast cancer.",
                    "Immunotherapy has shown promising results in triple-negative breast cancer.",
                    "Genomic testing helps personalize treatment approaches."
                ],
                constraints=[
                    Constraint(constraint_type="source_citation", required=True),
                    Constraint(constraint_type="avoid_guarantees", required=True),
                    Constraint(constraint_type="acknowledge_limitations", required=True)
                ],
                invalid_claims=[
                    "All breast cancers respond to the same treatment",
                    "Targeted therapies work for all types of breast cancer",
                    "New treatments guarantee a cure for all patients"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="HER2-targeted therapies such as trastuzumab (Herceptin), pertuzumab (Perjeta), and trastuzumab emtansine (T-DM1, Kadcyla) have significantly improved outcomes for patients with HER2-positive breast cancer. For hormone receptor-positive breast cancer, CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) in combination with endocrine therapy have shown substantial benefits in metastatic disease. Immunotherapy, particularly with immune checkpoint inhibitors like pembrolizumab, has shown promising results in triple-negative breast cancer. Genomic testing helps identify patients who may or may not benefit from chemotherapy.",
                    source="American Society of Clinical Oncology (ASCO) Guidelines 2023"
                ),
                ReferenceContent(
                    content="Targeted therapies for breast cancer include HER2-directed agents for HER2-positive disease, which have dramatically improved survival rates. CDK4/6 inhibitors have become standard of care for hormone receptor-positive metastatic breast cancer. For triple-negative breast cancer, immunotherapy combined with chemotherapy has shown improved outcomes in PD-L1-positive disease. PARP inhibitors have shown benefit in BRCA-mutated breast cancers. These advances highlight the importance of molecular testing to guide personalized treatment approaches.",
                    source="National Comprehensive Cancer Network (NCCN) Breast Cancer Guidelines"
                )
            ],
            difficulty=4,
            tags=["cancer", "oncology", "targeted therapy", "precision medicine"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_ERROR,
                HallucinationType.ENTITY_ERROR,
                HallucinationType.FACTUAL_FABRICATION
            ]
        ),
        
        # Medical edge case - COVID-19 with high potential for hallucination
        TestCase(
            id="med-003",
            prompt="What is the effectiveness of various COVID-19 vaccines, and how do they compare? Include details about long-term protection.",
            domain=Domain.MEDICAL,
            ground_truth=GroundTruth(
                facts=[
                    "mRNA vaccines (Pfizer-BioNTech, Moderna) showed ~95% efficacy in initial clinical trials.",
                    "Viral vector vaccines (Johnson & Johnson, AstraZeneca) showed lower initial efficacy rates.",
                    "Vaccine efficacy has varied against different variants.",
                    "Long-term protection data is still emerging and being studied."
                ],
                constraints=[
                    Constraint(constraint_type="source_citation", required=True),
                    Constraint(constraint_type="avoid_guarantees", required=True),
                    Constraint(constraint_type="acknowledge_uncertainty", required=True),
                    Constraint(constraint_type="time_bound_claims", required=True)
                ],
                invalid_claims=[
                    "Vaccines provide permanent immunity against COVID-19",
                    "All vaccines are equally effective against all variants",
                    "Specific percentages of long-term efficacy beyond current data",
                    "Claims of permanent immunity"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="In the initial clinical trials, the Pfizer-BioNTech and Moderna mRNA vaccines showed approximately 95% efficacy against symptomatic COVID-19. The Johnson & Johnson and AstraZeneca viral vector vaccines showed efficacy rates of around 70%. However, these rates were measured before the emergence of variants like Delta and Omicron. Studies have shown varying levels of effectiveness against different variants, with some reduction in protection. Booster doses have been shown to enhance protection. Data on long-term protection is still being collected, with studies indicating some waning of immunity over time, though protection against severe disease appears more durable.",
                    source="CDC COVID-19 Vaccine Effectiveness Research"
                ),
                ReferenceContent(
                    content="The efficacy of COVID-19 vaccines varies by type, timing, and variant. mRNA vaccines initially showed higher efficacy (approximately 95%) compared to viral vector vaccines (approximately 66-70%). All authorized vaccines have demonstrated strong protection against severe disease, hospitalization, and death. Protection against infection and symptomatic disease may decrease over time, particularly against newer variants. Long-term studies are ongoing, but current data suggests that while antibody levels may decline, memory B cells and T cell responses persist, potentially providing enduring protection against severe outcomes.",
                    source="WHO COVID-19 Vaccine Tracker and Database"
                )
            ],
            difficulty=5,
            tags=["COVID-19", "vaccines", "efficacy", "long-term", "variants"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.NUMERICAL_ERROR,
                HallucinationType.TEMPORAL_ERROR,
                HallucinationType.CONFIDENCE_MISMATCH
            ]
        )
    ]


# Legal domain test cases
def get_legal_test_cases() -> List[TestCase]:
    """Get legal domain test cases."""
    return [
        # Basic legal case - contract elements
        TestCase(
            id="leg-001",
            prompt="What are the essential elements required for a valid contract?",
            domain=Domain.LEGAL,
            ground_truth=GroundTruth(
                facts=[
                    "A valid contract generally requires offer, acceptance, consideration, legal capacity, and legal purpose.",
                    "Consideration refers to something of value exchanged between parties.",
                    "Capacity means parties must be legally capable of entering into contracts.",
                    "Some contracts may need to be in writing under the Statute of Frauds."
                ],
                constraints=[
                    Constraint(constraint_type="avoid_jurisdiction_claims", required=True),
                    Constraint(constraint_type="acknowledge_variations", required=True)
                ],
                invalid_claims=[
                    "All contracts must be in writing to be valid",
                    "Consideration must always be monetary",
                    "Requirements are identical in all legal jurisdictions"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="The essential elements of a valid contract generally include: (1) Offer: A clear, definite proposal to create specific terms of agreement; (2) Acceptance: Unequivocal agreement to the exact terms of the offer; (3) Consideration: Something of value exchanged between parties; (4) Legal capacity: Parties must be legally capable of entering contracts (e.g., not minors, not mentally incapacitated); (5) Legal purpose: The contract must be for a legal purpose and not against public policy. In addition, some contracts fall under the Statute of Frauds and must be in writing to be enforceable, such as contracts for the sale of land, contracts that cannot be performed within one year, and contracts for the sale of goods over a certain value.",
                    source="Restatement (Second) of Contracts"
                ),
                ReferenceContent(
                    content="A contract requires offer, acceptance, consideration, capacity of parties, and lawful object. An offer is a proposal that shows willingness to enter a legally binding agreement under specific terms. Acceptance must be clear, unequivocal, and generally mirror the terms of the offer. Consideration is something of value exchanged between the parties—it need not be money but must have some value. Capacity means parties must be of legal age and sound mind. The contract's purpose must be legal—contracts for illegal activities are void. While many contracts can be oral, certain types require written documentation under the Statute of Frauds.",
                    source="Black's Law Dictionary, Contract Elements"
                )
            ],
            difficulty=2,
            tags=["contracts", "legal elements", "consideration"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_ERROR,
                HallucinationType.CONTEXT_DRIFT
            ]
        ),
        
        # Complex legal case - intellectual property
        TestCase(
            id="leg-002",
            prompt="Explain the differences between patents, trademarks, and copyrights, including what they protect and for how long.",
            domain=Domain.LEGAL,
            ground_truth=GroundTruth(
                facts=[
                    "Patents protect inventions and novel processes for 20 years from filing date.",
                    "Trademarks protect brand identifiers like logos and can last indefinitely with proper renewal.",
                    "Copyrights protect original creative works for the author's life plus 70 years (for individual works).",
                    "Patents require novelty, non-obviousness, and utility."
                ],
                constraints=[
                    Constraint(constraint_type="source_citation", required=True),
                    Constraint(constraint_type="avoid_jurisdiction_claims", required=True),
                    Constraint(constraint_type="acknowledge_variations", required=True)
                ],
                invalid_claims=[
                    "All intellectual property protections last the same amount of time",
                    "Patents, trademarks, and copyrights protect the same things",
                    "Obtaining protection is automatic for all types"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Patents protect inventions and novel processes, granting the inventor exclusive rights for a limited period, typically 20 years from the filing date in the United States. To obtain a patent, an invention must be novel, non-obvious, and useful. Trademarks protect brand identifiers such as logos, names, and slogans that distinguish goods or services in the marketplace. With proper maintenance and renewal, trademarks can last indefinitely. Copyrights protect original works of authorship including literary, dramatic, musical, artistic, and certain other intellectual works. For works created by individuals after 1977, copyright protection lasts for the author's life plus 70 years. For works created anonymously, pseudonymously, or for hire, the copyright lasts 95 years from publication or 120 years from creation, whichever is shorter.",
                    source="U.S. Patent and Trademark Office Guide to Intellectual Property"
                ),
                ReferenceContent(
                    content="The three main types of intellectual property protection have distinct purposes and durations. Patents protect inventions, manufacturing processes, and technical designs, generally lasting 20 years from the application filing date. Trademarks protect brand identifiers like business names, logos, and slogans that distinguish one company's goods or services from another's. Trademarks can last indefinitely as long as they remain in use and are properly renewed (typically every 10 years). Copyrights protect original creative works including literature, music, art, film, and software. Copyright duration varies by jurisdiction but commonly lasts for the author's lifetime plus 70 years for individual works, or 95-120 years for corporate works.",
                    source="World Intellectual Property Organization (WIPO) Handbook"
                )
            ],
            difficulty=3,
            tags=["intellectual property", "patents", "trademarks", "copyrights", "duration"],
            expected_hallucination_types=[
                HallucinationType.TEMPORAL_ERROR,
                HallucinationType.FACTUAL_ERROR
            ]
        ),
        
        # Legal edge case - international law principles
        TestCase(
            id="leg-003",
            prompt="Explain the concept of 'universal jurisdiction' in international law, including examples of crimes that fall under it.",
            domain=Domain.LEGAL,
            ground_truth=GroundTruth(
                facts=[
                    "Universal jurisdiction allows states to prosecute certain serious crimes regardless of where they were committed.",
                    "Crimes typically subject to universal jurisdiction include genocide, crimes against humanity, war crimes, and torture.",
                    "The principle is based on the idea that some crimes are so heinous they affect the international community as a whole.",
                    "Application of universal jurisdiction varies significantly between countries."
                ],
                constraints=[
                    Constraint(constraint_type="source_citation", required=True),
                    Constraint(constraint_type="acknowledge_controversy", required=True),
                    Constraint(constraint_type="acknowledge_variations", required=True)
                ],
                invalid_claims=[
                    "Universal jurisdiction is uniformly accepted and applied globally",
                    "All international crimes fall under universal jurisdiction",
                    "Universal jurisdiction overrides all national sovereignty"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Universal jurisdiction is a legal principle that allows states to prosecute certain serious crimes regardless of where they were committed or the nationality of perpetrators or victims. Crimes typically subject to universal jurisdiction include genocide, crimes against humanity, war crimes, torture, and in some cases, terrorism and piracy. The principle is grounded in the concept that some crimes are so heinous that they affect the international community as a whole, and therefore any state has an interest in prosecuting them. Notable examples include Israel's prosecution of Adolf Eichmann, Spain's attempt to extradite Augusto Pinochet from the UK, and Belgium's universal jurisdiction law (later narrowed after diplomatic pressure). The application of universal jurisdiction varies significantly between countries and remains controversial, with some states viewing it as potential infringement on national sovereignty.",
                    source="Princeton Principles on Universal Jurisdiction"
                ),
                ReferenceContent(
                    content="Universal jurisdiction permits a state to exercise jurisdiction over certain international crimes even in the absence of a territorial, nationality, or other nexus to the state. The core crimes typically subject to universal jurisdiction are genocide, crimes against humanity, war crimes, and torture, as codified in conventions like the Geneva Conventions and the Convention Against Torture. Some states also extend universal jurisdiction to piracy, slavery, and certain acts of terrorism. The application of universal jurisdiction is not uniform across countries and has generated significant debate regarding its scope and limits. While some states have enacted broad universal jurisdiction laws, others have imposed various restrictions, such as requiring the presence of the accused in the territory or some connection to the prosecuting state. Notable cases include the Eichmann trial (Israel), the Pinochet case (Spain/UK), and prosecutions in countries like Belgium, Germany, and France for crimes committed in Rwanda, Syria, and elsewhere.",
                    source="International Justice Resource Center, Universal Jurisdiction"
                )
            ],
            difficulty=5,
            tags=["international law", "universal jurisdiction", "war crimes", "genocide"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.ENTITY_ERROR,
                HallucinationType.LOGICAL_INCONSISTENCY
            ]
        )
    ]


# Financial domain test cases
def get_financial_test_cases() -> List[TestCase]:
    """Get financial domain test cases."""
    return [
        # Basic financial case - investment types
        TestCase(
            id="fin-001",
            prompt="What are the main differences between stocks, bonds, and mutual funds as investment options?",
            domain=Domain.FINANCIAL,
            ground_truth=GroundTruth(
                facts=[
                    "Stocks represent ownership in a company and offer potential capital appreciation and dividends.",
                    "Bonds are debt securities that pay fixed interest and return principal at maturity.",
                    "Mutual funds pool money from multiple investors to purchase diversified portfolios of stocks, bonds, or other securities.",
                    "Stocks typically have higher risk and potential return compared to bonds."
                ],
                constraints=[
                    Constraint(constraint_type="include_risk_disclosure", required=True),
                    Constraint(constraint_type="avoid_guarantees", required=True)
                ],
                invalid_claims=[
                    "Stocks always outperform bonds",
                    "Mutual funds eliminate all investment risk",
                    "Bonds never lose value"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Stocks represent ownership in a company and offer potential capital appreciation and dividend income. They typically have higher risk and potential return than bonds. Bonds are debt securities where the investor lends money to an entity in exchange for fixed interest payments and the return of principal at maturity. They generally offer more stable income and lower risk than stocks but with lower potential returns. Mutual funds pool money from multiple investors to purchase diversified portfolios of stocks, bonds, or other securities, offering professional management and diversification, but charging management fees that can impact returns. The risk and return profile of a mutual fund depends on its underlying investments.",
                    source="Securities and Exchange Commission (SEC) Investor Education"
                ),
                ReferenceContent(
                    content="The three main investment vehicles have distinct characteristics. Stocks (equities) represent partial ownership in a company, with potential for capital appreciation and dividend income, but with significant price volatility. Bonds are debt instruments where investors lend money to the issuer in exchange for periodic interest payments and return of principal at maturity, offering generally lower risk and returns than stocks. Mutual funds pool assets from multiple investors to create diversified portfolios managed by professionals, which may include stocks, bonds, or both. Mutual funds offer diversification and professional management but charge fees that reduce returns. Each investment type has different risk-return profiles, with stocks typically being the most volatile with highest potential returns, bonds offering more stability with moderate returns, and mutual funds varying based on their composition.",
                    source="Financial Industry Regulatory Authority (FINRA) Investment Basics"
                )
            ],
            difficulty=1,
            tags=["investments", "stocks", "bonds", "mutual funds", "basics"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_ERROR,
                HallucinationType.CONFIDENCE_MISMATCH
            ]
        ),
        
        # Complex financial case - derivatives
        TestCase(
            id="fin-002",
            prompt="Explain options contracts, including the difference between calls and puts, and how they can be used in investment strategies.",
            domain=Domain.FINANCIAL,
            ground_truth=GroundTruth(
                facts=[
                    "Options are contracts giving the buyer the right, but not obligation, to buy or sell an underlying asset at a specified price before expiration.",
                    "Call options give the right to buy the underlying asset at the strike price.",
                    "Put options give the right to sell the underlying asset at the strike price.",
                    "Options can be used for hedging, income generation, and speculative strategies."
                ],
                constraints=[
                    Constraint(constraint_type="include_risk_disclosure", required=True),
                    Constraint(constraint_type="avoid_guarantees", required=True),
                    Constraint(constraint_type="distinguish_strategies", required=True)
                ],
                invalid_claims=[
                    "Options trading is risk-free",
                    "All options strategies generate consistent income",
                    "Options are suitable for all investors"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Options are contracts that give the buyer the right, but not the obligation, to buy or sell an underlying asset at a specified price (strike price) before a specified date (expiration). Call options give the holder the right to buy the underlying asset, while put options give the right to sell. Options can be used in various strategies: Hedging (protecting existing positions against adverse price movements), income generation (writing covered calls or cash-secured puts to collect premiums), speculation (betting on price movements with limited capital), and spread strategies (combining multiple options to create specific risk-reward profiles). Key components of options pricing include intrinsic value (difference between market price and strike price, if favorable) and time value (premium paid for the time remaining until expiration). Options involve substantial risk and are not suitable for all investors. Potential losses can exceed the initial investment for certain strategies, particularly when selling options.",
                    source="Options Clearing Corporation (OCC) Educational Materials"
                ),
                ReferenceContent(
                    content="Options contracts give holders rights regarding underlying assets without obligating them to exercise these rights. Call options provide the right to buy an underlying asset at the strike price before expiration, benefiting when prices rise above the strike price plus premium paid. Put options provide the right to sell an underlying asset at the strike price, benefiting when prices fall below the strike price minus premium paid. Common strategies include: Covered calls (owning the underlying asset and selling calls against it to generate income), protective puts (buying puts to hedge downside risk on owned assets), cash-secured puts (selling puts with cash reserved to purchase shares if assigned), and various spread strategies that combine multiple options positions. Options strategies can be categorized as bullish, bearish, or neutral, with different risk-reward characteristics. While options can limit risk in some cases, they can also create significant losses, especially when using leverage or complex strategies. Options trading requires understanding of time decay, volatility, and other factors affecting option prices.",
                    source="Chicago Board Options Exchange (CBOE) Options Institute"
                )
            ],
            difficulty=4,
            tags=["options", "derivatives", "calls", "puts", "investment strategies"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_ERROR,
                HallucinationType.LOGICAL_INCONSISTENCY,
                HallucinationType.NUMERICAL_ERROR
            ]
        ),
        
        # Financial edge case - cryptocurrency regulations
        TestCase(
            id="fin-003",
            prompt="What is the current regulatory status of cryptocurrencies in major global jurisdictions, and how does it affect their use as investments and payment methods?",
            domain=Domain.FINANCIAL,
            ground_truth=GroundTruth(
                facts=[
                    "Cryptocurrency regulations vary significantly across jurisdictions and continue to evolve.",
                    "Some countries have embraced cryptocurrencies with clear regulatory frameworks, while others have imposed restrictions or bans.",
                    "Regulatory considerations include securities laws, taxation, anti-money laundering (AML), and consumer protection.",
                    "Regulatory uncertainty affects institutional adoption, taxation, and legal status as payment methods."
                ],
                constraints=[
                    Constraint(constraint_type="time_bound_claims", required=True),
                    Constraint(constraint_type="acknowledge_uncertainty", required=True),
                    Constraint(constraint_type="acknowledge_variations", required=True),
                    Constraint(constraint_type="avoid_guarantees", required=True)
                ],
                invalid_claims=[
                    "Global regulations are uniform and clear",
                    "Specific predictions about future regulatory changes",
                    "Cryptocurrencies are universally legal or illegal"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Cryptocurrency regulation varies substantially across jurisdictions and continues to evolve rapidly. In the United States, cryptocurrencies fall under various regulatory regimes: the SEC generally considers many cryptocurrencies as securities, the CFTC has jurisdiction over crypto derivatives, and FinCEN applies anti-money laundering regulations to cryptocurrency exchanges. The European Union has developed the Markets in Crypto-Assets (MiCA) framework, establishing a comprehensive regulatory approach. Singapore has implemented a licensing framework for cryptocurrency service providers, while Japan recognizes cryptocurrencies as legal property. In contrast, China has banned cryptocurrency trading and mining. Regulatory considerations globally include securities classification, taxation, AML/KYC requirements, consumer protection, and use as payment methods. This regulatory diversity creates challenges for global operators and investors, affecting everything from tax treatment to the legality of using cryptocurrencies for payments. Institutional adoption is significantly influenced by regulatory clarity, with many traditional financial institutions hesitant to engage in markets without clear regulatory frameworks.",
                    source="International Monetary Fund (IMF) Fintech Notes: Regulation of Crypto Assets"
                ),
                ReferenceContent(
                    content="The regulatory landscape for cryptocurrencies remains fragmented globally. Major jurisdictions have taken divergent approaches: The United States applies a multi-agency approach with the SEC focusing on tokens deemed securities, the CFTC overseeing derivatives, and FinCEN and IRS addressing money transmission and taxation respectively. The European Union's MiCA regulation aims to create a harmonized framework across member states. The United Kingdom regulates through the FCA, focusing primarily on anti-money laundering but developing broader frameworks. Singapore and Switzerland have developed relatively crypto-friendly regulatory regimes with clear licensing requirements, while Japan has recognized cryptocurrencies as legal property with a registration system for exchanges. Some jurisdictions have taken restrictive approaches, with outright bans or severe limitations in countries like China, Turkey, and Bolivia. Regulatory uncertainty impacts institutional investment, taxation (with significant variations in capital gains treatment), and the practical use of cryptocurrencies as payment methods. Emerging regulatory trends include increased focus on stablecoins, DeFi regulation, environmental concerns related to mining, and central bank digital currencies (CBDCs) as alternatives to private cryptocurrencies.",
                    source="Global Legal Research Center, Cryptocurrency Regulations Worldwide"
                )
            ],
            difficulty=5,
            tags=["cryptocurrency", "regulation", "bitcoin", "blockchain", "securities", "compliance"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.TEMPORAL_ERROR,
                HallucinationType.CONTEXT_DRIFT
            ]
        )
    ]


# Technical domain test cases
def get_technical_test_cases() -> List[TestCase]:
    """Get technical domain test cases."""
    return [
        # Basic technical case - programming paradigms
        TestCase(
            id="tech-001",
            prompt="What are the main differences between object-oriented and functional programming paradigms?",
            domain=Domain.TECHNICAL,
            ground_truth=GroundTruth(
                facts=[
                    "Object-oriented programming organizes code around objects that encapsulate data and behavior.",
                    "Functional programming treats computation as the evaluation of mathematical functions and avoids changing state.",
                    "OOP emphasizes inheritance, encapsulation, and polymorphism.",
                    "Functional programming emphasizes immutability, first-class functions, and avoidance of side effects."
                ],
                constraints=[
                    Constraint(constraint_type="avoid_absolute_claims", required=True),
                    Constraint(constraint_type="cite_languages", required=True)
                ],
                invalid_claims=[
                    "One paradigm is always better than the other",
                    "The paradigms cannot be combined in programs",
                    "All modern languages support only one paradigm"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Object-oriented programming (OOP) organizes code around objects that combine data and behavior. It emphasizes concepts like encapsulation (bundling data with methods), inheritance (creating hierarchical relationships), and polymorphism (allowing objects to take different forms). Examples of OOP languages include Java, C++, and Python. Functional programming (FP) treats computation as the evaluation of mathematical functions and avoids changing state and mutable data. It emphasizes immutability, first-class and higher-order functions, pure functions (avoiding side effects), and declarative rather than imperative programming. Examples of functional languages include Haskell, Lisp, and Erlang. Many modern languages like JavaScript, Python, and Scala support both paradigms to varying degrees. The choice between paradigms depends on the problem domain, with OOP often favored for modeling real-world entities and FP for data processing, concurrency, and mathematical computations.",
                    source="Comparison of Programming Paradigms, IEEE Computer Society Journal"
                ),
                ReferenceContent(
                    content="The two major programming paradigms differ fundamentally in their approach to program structure. Object-oriented programming (OOP) is built around the concept of objects that contain data fields (attributes) and procedures (methods). Key principles include encapsulation (hiding internal state), inheritance (building hierarchies), and polymorphism (same interface for different types). OOP languages include Java, C#, and Ruby. Functional programming (FP) is based on the concept of mathematical functions and avoiding state changes. Key concepts include first-class and higher-order functions, pure functions, immutability, and recursion rather than iterative loops. FP languages include Haskell, Clojure, and F#. While historically separated, many modern languages incorporate elements of both paradigms. OOP tends to excel in modeling complex systems with many interacting components, while FP can simplify concurrency, testing, and reasoning about code by minimizing side effects. Neither paradigm is universally superior; rather, they represent different tools for different problems.",
                    source="ACM Computing Surveys, Programming Paradigms"
                )
            ],
            difficulty=2,
            tags=["programming", "paradigms", "object-oriented", "functional"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_ERROR,
                HallucinationType.LOGICAL_INCONSISTENCY
            ]
        ),
        
        # Complex technical case - machine learning
        TestCase(
            id="tech-002",
            prompt="Explain the differences between supervised, unsupervised, and reinforcement learning in machine learning, including example applications for each.",
            domain=Domain.TECHNICAL,
            ground_truth=GroundTruth(
                facts=[
                    "Supervised learning trains on labeled data to predict outputs from inputs, used in classification and regression tasks.",
                    "Unsupervised learning finds patterns in unlabeled data, used in clustering and dimensionality reduction.",
                    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors, used in game playing and robotics.",
                    "Each approach has different data requirements and suitable application domains."
                ],
                constraints=[
                    Constraint(constraint_type="cite_examples", required=True),
                    Constraint(constraint_type="avoid_oversimplification", required=True)
                ],
                invalid_claims=[
                    "One approach is always better than others",
                    "These categories never overlap in practice",
                    "Specific accuracy percentages without context"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Supervised learning trains models on labeled data to predict outputs from inputs. The algorithm learns a mapping function from input to output variables, with immediate feedback through error correction. Common applications include email spam detection (classification), housing price prediction (regression), and image recognition. Algorithms include linear regression, logistic regression, support vector machines, and neural networks. Unsupervised learning finds patterns in unlabeled data without predefined outputs. It identifies inherent structures in data, commonly used for clustering (grouping similar data points) and dimensionality reduction. Applications include customer segmentation, anomaly detection, and feature learning. Algorithms include k-means clustering, hierarchical clustering, and principal component analysis. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors and penalizing undesired ones. Unlike supervised learning, it doesn't require labeled input/output pairs but learns from interaction with an environment. Applications include game playing (AlphaGo), autonomous vehicles, robotics, and resource management. Algorithms include Q-learning, Deep Q Networks (DQN), and policy gradient methods.",
                    source="Introduction to Machine Learning, MIT Press"
                ),
                ReferenceContent(
                    content="The three main types of machine learning differ in their learning approach and applications. Supervised learning uses labeled training data with input-output pairs, providing a learning signal that guides the algorithm toward correct predictions. It's widely used in classification tasks (spam filtering, medical diagnosis), regression (price prediction, demand forecasting), and sequence generation tasks (machine translation). Common algorithms include decision trees, neural networks, and support vector machines. Unsupervised learning works with unlabeled data, identifying patterns without predefined outputs. Applications include clustering (market segmentation, social network analysis), dimensionality reduction (feature extraction, visualization), and anomaly detection (fraud detection, system health monitoring). Common algorithms include k-means, hierarchical clustering, autoencoders, and Gaussian mixture models. Reinforcement learning involves an agent learning to make decisions by performing actions in an environment to maximize cumulative rewards. Unlike supervised learning, it learns through trial and error rather than labeled examples. Applications include game playing (chess, poker), autonomous navigation, industrial automation, and recommendation systems. Common algorithms include Q-learning, SARSA, and various policy gradient methods. While these categories are distinct, modern systems often combine elements of multiple approaches, such as semi-supervised learning (combining supervised and unsupervised) and self-supervised learning (creating supervisory signals from unlabeled data).",
                    source="Machine Learning Fundamentals, University of California Berkeley"
                )
            ],
            difficulty=3,
            tags=["machine learning", "AI", "supervised learning", "unsupervised learning", "reinforcement learning"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_ERROR,
                HallucinationType.ENTITY_ERROR
            ]
        ),
        
        # Technical edge case - quantum computing
        TestCase(
            id="tech-003",
            prompt="Explain quantum computing, including quantum bits, quantum gates, and quantum supremacy. What are the current limitations and potential applications?",
            domain=Domain.TECHNICAL,
            ground_truth=GroundTruth(
                facts=[
                    "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously due to superposition.",
                    "Quantum gates manipulate qubits through operations like the Hadamard, CNOT, and Pauli gates.",
                    "Quantum supremacy refers to demonstrating a quantum computer's ability to solve problems intractable for classical computers.",
                    "Current limitations include qubit coherence time, error rates, and scalability challenges."
                ],
                constraints=[
                    Constraint(constraint_type="avoid_oversimplification", required=True),
                    Constraint(constraint_type="time_bound_claims", required=True),
                    Constraint(constraint_type="acknowledge_uncertainty", required=True)
                ],
                invalid_claims=[
                    "Quantum computers can solve all NP-hard problems efficiently",
                    "Quantum supremacy has made classical computers obsolete",
                    "Specific predictions about future quantum computing capabilities",
                    "Claims that quantum effects are not essential to quantum computing"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Quantum computing leverages quantum mechanical phenomena to perform computations. Quantum bits or qubits, unlike classical bits, can exist in superposition (being in multiple states simultaneously) and exhibit entanglement (where qubits become correlated and the state of one instantly influences others regardless of distance). Quantum gates are operations that manipulate qubits, including single-qubit gates like the Hadamard (creating superposition) and Pauli gates, and multi-qubit gates like CNOT (controlled-NOT). Quantum supremacy refers to the demonstration that a quantum computer can solve a problem that is practically impossible for classical computers to solve in a reasonable timeframe. Google claimed to achieve this milestone in 2019 with its 53-qubit Sycamore processor. Current limitations include short coherence times (qubits maintaining their quantum state), high error rates requiring error correction which demands more physical qubits, and engineering challenges in scaling up qubit numbers while maintaining quality. Potential applications include cryptography (breaking current encryption and creating new quantum-secure methods), drug discovery and materials science (simulating molecular and quantum systems), optimization problems (supply chain, traffic flow), and machine learning enhancements.",
                    source="Introduction to Quantum Computing, MIT Quantum Information Science"
                ),
                ReferenceContent(
                    content="Quantum computing harnesses quantum mechanical principles to process information. The fundamental unit, the qubit, can exist in states |0⟩, |1⟩, or a superposition of both simultaneously. Multiple qubits can be entangled, creating correlations impossible in classical systems. Quantum algorithms manipulate qubits using quantum gates such as the Hadamard (creating superposition), Pauli-X (quantum equivalent of classical NOT), and CNOT (creating entanglement between qubits). These gates form quantum circuits that perform computation. Quantum supremacy (or quantum advantage) refers to demonstrating that a quantum computer can solve a problem substantially faster than the best known algorithms running on classical computers. Notable claims include Google's 2019 experiment on a sampling problem and more recent demonstrations by Chinese researchers. Current quantum computers face significant limitations: decoherence (quantum states degrading due to environmental interaction), requiring operation at near absolute zero temperatures, high error rates necessitating quantum error correction, and scalability challenges in increasing qubit count while maintaining quality. Promising near-term applications include quantum chemistry simulations, optimization problems (particularly through quantum annealing), certain machine learning tasks, and eventually, Shor's algorithm for factoring large numbers (threatening current cryptographic systems). Industries actively exploring quantum computing include pharmaceuticals, materials science, finance, logistics, and cybersecurity.",
                    source="Quantum Computing: Progress and Prospects, National Academies of Sciences, Engineering, and Medicine"
                )
            ],
            difficulty=5,
            tags=["quantum computing", "qubits", "quantum supremacy", "quantum gates"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.NUMERICAL_ERROR,
                HallucinationType.CONFIDENCE_MISMATCH
            ]
        )
    ]


# Edge cases explicitly designed to test hallucination detection
def get_edge_test_cases() -> List[TestCase]:
    """Get edge cases for hallucination testing."""
    return [
        # Mixture of factual and non-factual content
        TestCase(
            id="edge-001",
            prompt="Describe the history and development of the Internet, including key milestones and inventors.",
            domain=Domain.EDGE_CASE,
            ground_truth=GroundTruth(
                facts=[
                    "ARPANET, developed in the late 1960s, was a precursor to the Internet.",
                    "TCP/IP protocol was developed in the 1970s by Vint Cerf and Bob Kahn.",
                    "Tim Berners-Lee invented the World Wide Web in 1989-1991 at CERN.",
                    "Commercialization and widespread adoption occurred in the 1990s."
                ],
                constraints=[
                    Constraint(constraint_type="source_citation", required=True),
                    Constraint(constraint_type="avoid_fabrications", required=True)
                ],
                invalid_claims=[
                    "The Internet was invented by a single person",
                    "The Internet was originally created for public social networking",
                    "Fictional inventors or non-existent projects",
                    "False specific dates that contradict historical records"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="The Internet's development began with ARPANET (Advanced Research Projects Agency Network), funded by the U.S. Department of Defense in the late 1960s. The first ARPANET connection was established between UCLA and Stanford Research Institute in 1969. In the 1970s, Vinton Cerf and Robert Kahn developed the Transmission Control Protocol and Internet Protocol (TCP/IP), which standardized how data could be transmitted between networks. TCP/IP was adopted by ARPANET in 1983, marking a critical milestone. In 1989-1991, Tim Berners-Lee, while working at CERN, invented the World Wide Web, creating the first web browser, server, and web pages. The National Science Foundation's NSFNET expanded Internet infrastructure in the 1980s and early 1990s. Commercialization accelerated in the 1990s after the privatization of the Internet backbone and the launch of graphical browsers like Mosaic (1993) and Netscape Navigator (1994). The Internet's growth exploded with the dot-com boom of the late 1990s, transforming from primarily academic and research use to mainstream adoption.",
                    source="Internet Society, Brief History of the Internet"
                ),
                ReferenceContent(
                    content="The Internet evolved from early networking research funded by DARPA (Defense Advanced Research Projects Agency). Key milestones include: 1969: First ARPANET connection established between UCLA and Stanford Research Institute. 1971: Ray Tomlinson invented email. 1974: TCP protocol specification published by Vint Cerf and Bob Kahn. 1978: TCP split into TCP/IP. 1983: ARPANET transitions to TCP/IP protocols, often considered the birth of the Internet as we know it. 1984: Domain Name System (DNS) introduced. 1989-1991: Tim Berners-Lee invents the World Wide Web at CERN, creating HTTP, HTML, and the first browser and server software. 1991: NSFNET lifted restrictions on commercial use. 1993: Mosaic, the first popular graphical web browser, released. 1994-1995: Netscape Navigator and Internet Explorer launched, fueling browser competition. 1995-2000: Dot-com boom accelerates commercial development. 1998: Google founded. 2004-2010: Rise of social media and Web 2.0. The Internet's development was the work of many individuals, organizations, and governments rather than any single inventor, with different teams contributing critical components to its evolution.",
                    source="Computer History Museum, Internet History Program"
                )
            ],
            difficulty=3,
            tags=["internet", "history", "technology", "edge case"],
            expected_hallucination_types=[
                HallucinationType.ENTITY_ERROR,
                HallucinationType.TEMPORAL_ERROR,
                HallucinationType.FACTUAL_FABRICATION
            ]
        ),
        
        # Deliberately ambiguous prompt
        TestCase(
            id="edge-002",
            prompt="Tell me about quantum effects in biological systems and their significance.",
            domain=Domain.EDGE_CASE,
            ground_truth=GroundTruth(
                facts=[
                    "Quantum effects have been proposed in some biological processes like photosynthesis and enzyme catalysis.",
                    "Quantum coherence may play a role in the efficiency of photosynthetic energy transfer.",
                    "Quantum tunneling has been implicated in certain enzyme reactions.",
                    "Research in quantum biology is ongoing with limited definitive conclusions."
                ],
                constraints=[
                    Constraint(constraint_type="acknowledge_uncertainty", required=True),
                    Constraint(constraint_type="avoid_oversimplification", required=True),
                    Constraint(constraint_type="source_citation", required=True)
                ],
                invalid_claims=[
                    "Quantum effects definitely explain consciousness",
                    "All biological processes rely on quantum effects",
                    "Quantum biology is a completely established field with consensus",
                    "Specific biological capabilities attributed to quantum effects without evidence"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Quantum biology examines quantum mechanical phenomena in biological systems. Several key processes have been investigated: Photosynthesis may use quantum coherence to achieve high efficiency in energy transfer, as suggested by studies observing long-lasting quantum coherence in light-harvesting complexes. Enzyme catalysis in some systems appears to involve quantum tunneling, where particles pass through energy barriers rather than over them, potentially explaining the high efficiency of certain enzyme reactions. Magnetoreception in birds may involve quantum entanglement in cryptochrome proteins, helping explain how birds detect Earth's magnetic field for navigation. There are also speculative proposals about quantum effects in olfaction (smell) and consciousness, though these remain controversial. Research in quantum biology faces challenges including maintaining quantum effects in warm, wet biological environments which typically cause decoherence. The field remains largely exploratory, with ongoing debate about the significance and extent of quantum effects in biological systems. While some quantum effects in biology are well-documented, their functional importance and evolutionary advantage continue to be investigated.",
                    source="Nature Reviews Physics, Quantum Biology: An Introduction"
                ),
                ReferenceContent(
                    content="The intersection of quantum mechanics and biology, known as quantum biology, examines how quantum phenomena might influence biological processes. Several biological systems have shown evidence of quantum effects: In photosynthesis, studies have suggested quantum coherence may contribute to the near-perfect efficiency of energy transfer in light-harvesting complexes, allowing energy to explore multiple pathways simultaneously. Enzyme-catalyzed reactions, particularly those involving hydrogen transfer, show evidence of quantum tunneling, where particles pass through energy barriers rather than over them. This may explain why some enzymes work more efficiently than classical physics would predict. Avian magnetoreception potentially utilizes quantum entanglement in cryptochrome proteins to detect Earth's magnetic field for navigation. Other areas of investigation include quantum effects in olfaction (the vibration theory of smell) and vision (where rhodopsin may detect single photons through quantum processes). The field remains controversial, with ongoing debate about whether quantum effects are functionally significant or merely incidental. A key challenge is that quantum coherence typically requires isolated, cold environments, while biological systems are warm and wet. Research continues to determine whether biological systems have evolved mechanisms to protect or utilize quantum effects despite these challenging conditions.",
                    source="Proceedings of the Royal Society B, Quantum Effects in Biological Systems"
                )
            ],
            difficulty=4,
            tags=["quantum biology", "photosynthesis", "enzyme catalysis", "ambiguous", "edge case"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.CONFIDENCE_MISMATCH,
                HallucinationType.CONTEXT_DRIFT
            ]
        ),
        
        # Contradictory reference materials
        TestCase(
            id="edge-003",
            prompt="Explain the health benefits and risks of coffee consumption according to current research.",
            domain=Domain.EDGE_CASE,
            ground_truth=GroundTruth(
                facts=[
                    "Moderate coffee consumption has been associated with reduced risk of certain conditions like Parkinson's disease and type 2 diabetes.",
                    "Coffee consumption may increase blood pressure in some individuals, especially those sensitive to caffeine.",
                    "Research shows mixed results regarding coffee's effects on different health outcomes.",
                    "Individual factors affect how coffee impacts health."
                ],
                constraints=[
                    Constraint(constraint_type="acknowledge_uncertainty", required=True),
                    Constraint(constraint_type="present_multiple_perspectives", required=True),
                    Constraint(constraint_type="source_citation", required=True)
                ],
                invalid_claims=[
                    "Coffee definitively cures or causes specific diseases",
                    "All research conclusively agrees on coffee's effects",
                    "Specific recommended amounts that apply to everyone"
                ]
            ),
            reference_content=[
                ReferenceContent(
                    content="Coffee consumption offers numerous health benefits according to recent research. Regular coffee drinkers show reduced risk of several neurodegenerative conditions, including Parkinson's disease (25-30% lower risk) and Alzheimer's disease. Multiple studies indicate coffee consumption reduces type 2 diabetes risk by up to 30% for heavy coffee drinkers. Coffee contains powerful antioxidants that combat inflammation and may reduce certain cancer risks, particularly liver and colorectal cancers. The caffeine in coffee appears to improve cognitive function, increase alertness, and may enhance long-term memory. Regular coffee consumption has been linked to a moderate decrease in all-cause mortality in large population studies. However, some individuals should exercise caution: pregnant women are advised to limit caffeine consumption, those with sleep disorders may experience worsened symptoms, and individuals with certain heart conditions may experience increased blood pressure or irregular heartbeats with caffeine consumption.",
                    source="Harvard School of Public Health, Coffee and Health Report"
                ),
                ReferenceContent(
                    content="Coffee consumption presents significant health concerns according to some studies. The caffeine in coffee can cause or worsen anxiety, jitteriness, and sleep disturbances in many individuals. Regular consumption leads to physical dependence, with withdrawal symptoms when intake stops. Coffee consumption has been linked to increased blood pressure, particularly in people already at risk for hypertension. Certain brewing methods result in cafestol and kahweol compounds that can raise LDL cholesterol levels. Unfiltered coffee may increase homocysteine levels, a risk factor for heart disease. Heavy caffeine intake has been associated with increased risk of miscarriage and low birth weight in pregnant women. Coffee can worsen symptoms of gastroesophageal reflux disease (GERD) and irritable bowel syndrome in susceptible individuals. Some studies suggest excessive coffee consumption may accelerate bone loss in individuals at risk for osteoporosis. While moderate consumption may be harmless for many, individual sensitivity varies significantly based on genetics, pre-existing conditions, and other factors.",
                    source="Journal of Caffeine Research, Meta-analysis on Health Effects of Coffee"
                )
            ],
            difficulty=3,
            tags=["coffee", "health", "contradictory", "edge case"],
            expected_hallucination_types=[
                HallucinationType.FACTUAL_FABRICATION,
                HallucinationType.NUMERICAL_ERROR,
                HallucinationType.LOGICAL_INCONSISTENCY
            ]
        )
    ]


def get_all_test_cases() -> List[TestCase]:
    """Get all test cases from all domains."""
    all_cases = []
    all_cases.extend(get_medical_test_cases())
    all_cases.extend(get_legal_test_cases())
    all_cases.extend(get_financial_test_cases())
    all_cases.extend(get_technical_test_cases())
    all_cases.extend(get_edge_test_cases())
    return all_cases


def get_test_cases_by_domain(domain: Domain) -> List[TestCase]:
    """Get test cases for a specific domain."""
    if domain == Domain.MEDICAL:
        return get_medical_test_cases()
    elif domain == Domain.LEGAL:
        return get_legal_test_cases()
    elif domain == Domain.FINANCIAL:
        return get_financial_test_cases()
    elif domain == Domain.TECHNICAL:
        return get_technical_test_cases()
    elif domain == Domain.EDGE_CASE:
        return get_edge_test_cases()
    else:
        return []


def get_test_case_by_id(case_id: str) -> Optional[TestCase]:
    """Get a test case by its ID."""
    for case in get_all_test_cases():
        if case.id == case_id:
            return case
    return None