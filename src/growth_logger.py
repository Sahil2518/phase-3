import os
import json
import logging
from datetime import datetime
import uuid
import sys

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task06_system.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GrowthLogger:
    """
    Handles structured JSON logging for growth tracking, capturing
    impressions, clicks, applies, and shortlists with exact positions
    and model version metadata.
    
    This ensures we can join outcomes back to impressions.
    """
    
    def __init__(self, log_path: str = "logs/growth_events.jsonl"):
        """
        Initializes the GrowthLogger.

        Parameters
        ----------
        log_path : str
            The path where JSON lines events will be appended.
        """
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._ensure_file()
        
    def _ensure_file(self):
        """Ensures the target log file exists and is writable."""
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                pass
        except Exception as e:
            logger.critical(f"Cannot write to growth log path: {self.log_path}. Error: {e}")
            raise

    def log_impression(self, impression_id: str, session_id: str, candidate_id: str, model_version: str, ranked_items: list):
        """
        Logs a full ranked list impression.

        Parameters
        ----------
        impression_id : str
            Unique ID for this ranked list presentation.
        session_id : str
            User session ID.
        candidate_id : str
            ID of the user viewing the list.
        model_version : str
            The model version used to produce the ranking.
        ranked_items : list of dict
            List of item dictionaries, each containing 'job_id', 'position', and 'score'.
        """
        if not impression_id or not model_version:
            logger.error("Failed to log impression: Missing impression_id or model_version.")
            return
            
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "impression",
            "timestamp": datetime.utcnow().isoformat(),
            "impression_id": impression_id,
            "session_id": session_id,
            "candidate_id": candidate_id,
            "model_version": model_version,
            "items": ranked_items
        }
        self._write_event(event)

    def log_interaction(self, event_type: str, impression_id: str, job_id: str, position: int, session_id: str = None):
        """
        Logs an interaction event (click, apply, shortlist) tied to a previous impression.

        Parameters
        ----------
        event_type : str
            The type of interaction ('click', 'apply', 'shortlist').
        impression_id : str
            The impression ID this interaction originated from.
        job_id : str
            The ID of the interacted job.
        position : int
            The ranking position the job was shown at.
        session_id : str, optional
            The user session ID.
        """
        valid_types = ["click", "apply", "shortlist"]
        if event_type not in valid_types:
            logger.warning(f"Invalid interaction event_type '{event_type}'. Must be one of {valid_types}.")
            return
            
        if not impression_id:
            logger.error(f"Cannot log {event_type} without an impression_id.")
            return

        if position < 1:
            logger.warning(f"Logging interaction with non-positive position ({position}).")

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "impression_id": impression_id,
            "session_id": session_id,
            "job_id": job_id,
            "position": position
        }
        self._write_event(event)

    def _write_event(self, event_dict: dict):
        """
        Thread-safe (in context of basic file appends) writing of JSONL.
        
        Parameters
        ----------
        event_dict : dict
            The event dictionary to write.
        """
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_dict) + "\n")
        except Exception as e:
            logger.error(f"Failed to write event to log: {e}")
