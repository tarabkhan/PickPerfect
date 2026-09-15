DEFAULT_WEIGHTS = {
    "popular": 0.2,
    "ncf": 0.4,
    "content": 0.4,
}

# Global references to recommendation scoring functions
_popularity_fn = None
_ncf_score_fn = None
_content_score_fn = None


def configure(popular_fn=None, ncf_fn=None, content_fn=None):
    """
    Registers the scoring functions used to compute individual component scores.
    Pass functions that return a normalized score between 0.0 and 1.0.
    """
    global _popularity_fn, _ncf_score_fn, _content_score_fn

    _popularity_fn = popular_fn
    _ncf_score_fn = ncf_fn
    _content_score_fn = content_fn


def compute_ai_score(product, customer_id=None, reference_product=None):
    """
    Computes a 0-100 AI Personalization Score for a product based on available signals.
    """
    components = {
        "popular": _popularity_fn(product) if callable(_popularity_fn) else None,
        "ncf": _ncf_score_fn(customer_id, product) if (customer_id and callable(_ncf_score_fn)) else None,
        "content": _content_score_fn(reference_product, product) if (reference_product and callable(_content_score_fn)) else None,
    }

    # Filter out missing or None component scores
    available = {k: v for k, v in components.items() if v is not None}
    total_weight = sum(DEFAULT_WEIGHTS[k] for k in available)

    fallback_used = False
    weights_used = {}
    final_score = None

    if total_weight > 0:
        # Re-normalize weights so available components sum up to 100%
        weighted_sum = sum(DEFAULT_WEIGHTS[k] * v for k, v in available.items())
        final_score = (weighted_sum / total_weight) * 100
        weights_used = {k: round(DEFAULT_WEIGHTS[k] / total_weight, 4) for k in available}
    elif callable(_popularity_fn):
        fallback_score = _popularity_fn(product)
        if fallback_score is not None:
            final_score = fallback_score * 100
            fallback_used = True

    result = dict(components)
    result["final_ai_score"] = round(final_score) if final_score is not None else 0
    result["weights_used"] = weights_used
    result["fallback_used"] = fallback_used

    return result