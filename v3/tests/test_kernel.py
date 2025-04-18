"""
Integration test for ACDEKKernel orchestration of enrichment, constraint, and confidence stages.
"""
from acdek.acdek_kernel import ACDEKKernel
from types import SimpleNamespace

def test_full_pipeline_monkeypatched():
    # Prepare dummy components
    dummy_enrich = SimpleNamespace(
        success=True,
        data={'val': 1},
        errors={},
        metrics={'t': 0},
        stage_results={'stage': {}}
    )
    dummy_constraint = SimpleNamespace(
        to_dict=lambda: {'passed': True, 'mode': 'strict', 'results': [], 'violations': [], 'metrics': {}, 'timestamp': 0},
        passed=True
    )
    dummy_conf = SimpleNamespace(
        aggregate_score=0.75,
        factors={'vector_quality': 0.0},
        metadata={'info': 'test'},
        timestamp=0
    )
    # Instantiate kernel and inject dummies
    kernel = ACDEKKernel()
    kernel.enrichment_pipeline = SimpleNamespace(execute=lambda inp, ctx: dummy_enrich)
    kernel.constraint_verifier = SimpleNamespace(validate=lambda data: dummy_constraint)
    kernel.confidence_scorer = SimpleNamespace(score=lambda data: dummy_conf)
    # Run pipeline
    result = kernel.run({'input': 'data'})
    # Validate enrichment section
    assert 'enrichment' in result
    enrich = result['enrichment']
    assert enrich['success'] is True
    assert enrich['data'] == {'val': 1}
    assert enrich.get('fallback_depth') == 0
    assert enrich.get('fallback_penalty') == 0.0 or enrich.get('fallback_penalty') == 0
    # Validate constraints
    assert 'constraints' in result
    assert result['constraints']['passed'] is True
    # Validate confidence
    assert 'confidence' in result
    assert isinstance(result['confidence']['aggregate_score'], float)
    assert isinstance(result['confidence']['metadata'], dict)
    # Validate timing data
    assert 'timing' in result
    for key in ('enrichment', 'constraint', 'confidence', 'total'):
        assert key in result['timing']
        assert isinstance(result['timing'][key], float)