from analyzer import QuantitativeRiskAnalyzer

def test_calculate_statistics():
    data = [10.0, 12.0, 23.0, 23.0, 16.0, 21.0, 30.0]
    analyzer = QuantitativeRiskAnalyzer(data)
    stats = analyzer.calculate_statistics()
    
    assert "mean" in stats
    assert "std_dev" in stats
    assert stats["mean"] == 19.285714285714285

def test_calculate_var():
    data = [0.010, -0.020, 0.015, -0.008, 0.004, -0.030, 0.012, -0.006]
    analyzer = QuantitativeRiskAnalyzer(data)
    var = analyzer.calculate_var(0.95)
    
    assert isinstance(var, float)
    assert var > 0