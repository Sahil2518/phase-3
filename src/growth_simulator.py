import os
import sys
import uuid
import random
import logging
from src.growth_logger import GrowthLogger

# Rule 2: Structured Logging
logger = logging.getLogger(__name__)

# Rule 5: Reproducibility
random.seed(42)

class GrowthSimulator:
    """
    Simulates traffic and candidate interactions to generate realistic
    growth telemetry volume. Evaluates funnel drop-offs (impression -> click -> apply -> shortlist).
    """

    def __init__(self, log_path: str = "logs/growth_events.jsonl", break_mode: bool = False):
        """
        Initializes the simulator.
        
        Parameters
        ----------
        log_path : str
            Path to write simulated logs.
        break_mode : bool
            If True, purposefully introduces errors (missing model_version, missing impression_id)
            to test graceful degradation.
        """
        self.logger = GrowthLogger(log_path=log_path)
        self.break_mode = break_mode
        self.model_version = "v1.0.0-lightgbm" if not break_mode else None # Intentional break

    def simulate_traffic(self, num_sessions: int = 1000, items_per_impression: int = 10):
        """
        Runs the simulation loop, generating events.
        
        Parameters
        ----------
        num_sessions : int
            Number of impressions to generate.
        items_per_impression : int
            Number of jobs to rank per impression.
        """
        logger.info(f"Starting traffic simulation. Sessions={num_sessions}, Break Mode={self.break_mode}")
        
        for _ in range(num_sessions):
            try:
                self._simulate_session(items_per_impression)
            except Exception as e:
                # Rule 2: Non-fatal step guards
                logger.error(f"Error simulating session: {e}")
                
        logger.info("Traffic simulation complete.")

    def _simulate_session(self, num_items: int):
        """
        Simulates a single candidate viewing a ranked list and taking action.
        
        Parameters
        ----------
        num_items : int
            Number of items in the list.
        """
        session_id = str(uuid.uuid4())
        candidate_id = f"c_{random.randint(1000, 9999)}"
        
        if self.break_mode and random.random() < 0.1:
            impression_id = None # Intentional break (10% of the time in break mode)
        else:
            impression_id = str(uuid.uuid4())

        # 1. Generate items and log impression
        ranked_items = []
        for pos in range(1, num_items + 1):
            # score decreases as position increases
            base_score = 0.95 - (0.05 * pos) + random.uniform(-0.02, 0.02)
            score = max(0.0, min(1.0, base_score))
            
            ranked_items.append({
                "job_id": f"j_{random.randint(10000, 99999)}",
                "position": pos,
                "score": round(score, 4)
            })

        self.logger.log_impression(
            impression_id=impression_id,
            session_id=session_id,
            candidate_id=candidate_id,
            model_version=self.model_version,
            ranked_items=ranked_items
        )

        # Skip interaction simulation if impression failed due to break_mode
        if not impression_id:
            return

        # 2. Simulate interactions with strong position bias
        for item in ranked_items:
            pos = item["position"]
            
            # Position bias: P(Click) drops exponentially with position
            click_prob = 0.3 * (0.6 ** (pos - 1))
            
            if random.random() < click_prob:
                self.logger.log_interaction("click", impression_id, item["job_id"], pos, session_id)
                
                # Apply probability given Click is ~30%
                apply_prob = 0.3
                if random.random() < apply_prob:
                    self.logger.log_interaction("apply", impression_id, item["job_id"], pos, session_id)
                    
                    # Shortlist probability given Apply is ~20%
                    shortlist_prob = 0.2
                    if random.random() < shortlist_prob:
                        self.logger.log_interaction("shortlist", impression_id, item["job_id"], pos, session_id)

if __name__ == "__main__":
    # Test execution
    simulator = GrowthSimulator()
    simulator.simulate_traffic(num_sessions=100)
