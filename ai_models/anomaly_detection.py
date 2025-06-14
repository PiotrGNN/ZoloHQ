class AnomalyDetector:
    def __init__(self, sensitivity=0.1):
        self.sensitivity = sensitivity
    def detect_anomalies(self, df):
        import pandas as pd
        # Dummy: return last 5 rows as anomalies
        return df.tail(5)
    def get_confidence(self):
        return 0.95
