import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Rule 2: Structured Logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

class PersonalizationSignals(BaseModel):
    skill_multiplier: float = Field(1.0, description="Multiplier for skill score")
    seniority_multiplier: float = Field(1.0, description="Multiplier for seniority score")

class UserContext(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    org_id: str = Field(..., description="Current authorized organization ID")
    signals: PersonalizationSignals = Field(default_factory=PersonalizationSignals, description="User's personal ML signals")

class IdentityManager:
    """
    Simulates a SCIM / SSO Identity Provider backend.
    Manages user provisioning and maintains strict org mappings to prevent cross-contamination.
    """
    def __init__(self):
        self._users: Dict[str, UserContext] = {}
        
    def provision_user(self, user_id: str, org_id: str, signals: Optional[PersonalizationSignals] = None):
        """Provisions a new user into an org with optional personalization signals."""
        if not signals:
            signals = PersonalizationSignals()
        self._users[user_id] = UserContext(user_id=user_id, org_id=org_id, signals=signals)
        logger.info(f"Provisioned user {user_id} into org {org_id}.")
        
    def deprovision_user(self, user_id: str):
        """Hard deletes a user and purges all signals to satisfy the worry check."""
        if user_id in self._users:
            del self._users[user_id]
            logger.info(f"De-provisioned user {user_id}. All signals purged.")
            
    def transfer_user(self, user_id: str, new_org_id: str):
        """
        Transfers a user to a new org.
        Their personal ML signals move with them, but their authorization scope strictly changes.
        """
        if user_id not in self._users:
            raise ValueError(f"Cannot transfer unknown user {user_id}.")
            
        old_org = self._users[user_id].org_id
        # Signals persist, org changes
        self._users[user_id].org_id = new_org_id
        logger.warning(f"Transferred user {user_id} from {old_org} to {new_org_id}. Signals retained.")
        
    def get_user_context(self, user_id: str) -> UserContext:
        """
        Source of Truth for User Auth and Signals.
        """
        if user_id not in self._users:
            raise PermissionError(f"User {user_id} is not recognized or is de-provisioned.")
        return self._users[user_id]
