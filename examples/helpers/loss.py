def improved(min: float, max: float) -> float:
    assert max >= min
    diff = max - min
    return diff / max * 100