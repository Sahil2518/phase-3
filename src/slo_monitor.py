import pandas as pd
import numpy as np
import logging
import sys

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SLOMonitor:
    """
    Evaluates inference telemetry data against defined SLOs and generates alerts.

    The monitor checks three key dimensions of model service health:
    1. Availability: % of non-5xx responses.
    2. Latency: p95 response time.
    3. Quality: Ensuring scores are not degenerate/constant.
    """
    
    def __init__(self, target_availability=0.999, target_p95_latency=200, min_score_variance=1e-4):
        self.target_availability = target_availability
        self.target_p95_latency = target_p95_latency
        self.min_score_variance = min_score_variance
        
    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluates the dataframe of logs against SLOs.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing telemetry logs.

        Returns
        -------
        report : dict
            Dictionary detailing the SLO evaluation results and any fired alerts.
        """
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty or None.")
            
        required_cols = ['latency_ms', 'http_status', 'prediction_score']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        total_requests = len(df)
        
        # 1. Availability
        successful_requests = df[~df['http_status'].isin([500, 502, 503, 504])]
        availability = len(successful_requests) / total_requests
        
        # 2. Latency
        p95_latency = np.percentile(df['latency_ms'], 95)
        
        # 3. Quality
        score_variance = np.var(df['prediction_score'])
        avg_score = np.mean(df['prediction_score'])
        
        alerts = []
        
        if availability < self.target_availability:
            alerts.append({
                "type": "AVAILABILITY_BREACH",
                "message": f"Availability dropped to {availability:.4%} (Target: >= {self.target_availability:.1%})"
            })
            
        if p95_latency > self.target_p95_latency:
            alerts.append({
                "type": "LATENCY_BREACH",
                "message": f"p95 Latency spiked to {p95_latency:.2f}ms (Target: <= {self.target_p95_latency}ms)"
            })
            
        if score_variance < self.min_score_variance:
            alerts.append({
                "type": "QUALITY_BREACH",
                "message": f"Model scores are degenerate/constant. Variance: {score_variance:.6f}. Avg Score: {avg_score:.4f}"
            })
            
        report = {
            "metrics": {
                "total_requests": total_requests,
                "availability": round(availability, 6),
                "p95_latency": round(p95_latency, 2),
                "score_variance": round(float(score_variance), 6)
            },
            "slos": {
                "target_availability": self.target_availability,
                "target_p95_latency": self.target_p95_latency,
            },
            "alerts_fired": len(alerts) > 0,
            "alerts": alerts
        }
        
        return report

def main():
    pass

if __name__ == "__main__":
    main()
