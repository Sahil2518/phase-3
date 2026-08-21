import json
import logging
import os
from pydantic import BaseModel, Field
from typing import Dict

logger = logging.getLogger(__name__)

class TenantConfig(BaseModel):
    """
    Contract for Tenant Model Configuration.
    Defines the thresholds and weights used for multi-tenant inference.
    """
    skill_weight: float = Field(0.5, description="Weight for skill match score.")
    location_weight: float = Field(0.3, description="Weight for location match score.")
    seniority_weight: float = Field(0.2, description="Weight for seniority match score.")
    match_threshold: float = Field(0.6, description="Minimum score to be considered a valid match.")

class TenantManager:
    """
    Manages loading and serving per-tenant configurations without code branches.
    """
    def __init__(self, config_file: str = "models/tenant_configs.json"):
        self.config_file = config_file
        self.configs: Dict[str, TenantConfig] = {}
        self._load_configs()

    def _load_configs(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    for tenant_id, cfg in data.items():
                        self.configs[tenant_id] = TenantConfig(**cfg)
                logger.info(f"Loaded {len(self.configs)} tenant configurations from {self.config_file}.")
            except Exception as e:
                logger.error(f"Failed to load tenant configs: {e}")
        else:
            logger.warning(f"Tenant config file {self.config_file} not found. Starting empty.")

    def save_configs(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        # Handle dict() vs model_dump() for pydantic compatibility
        data = {}
        for tid, cfg in self.configs.items():
            if hasattr(cfg, "model_dump"):
                data[tid] = cfg.model_dump()
            else:
                data[tid] = cfg.dict()
        try:
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved tenant configurations to {self.config_file}.")
        except Exception as e:
            logger.error(f"Failed to save tenant configs: {e}")

    def add_or_update_tenant(self, tenant_id: str, config: TenantConfig):
        """
        Adds or updates the configuration for a given tenant.
        
        Parameters
        ----------
        tenant_id : str
            The identifier for the tenant.
        config : TenantConfig
            The configuration parameters for the tenant.
        """
        self.configs[tenant_id] = config
        self.save_configs()

    def get_config(self, tenant_id: str) -> TenantConfig:
        """
        Retrieves the configuration for a specific tenant.
        
        Parameters
        ----------
        tenant_id : str
            The identifier for the tenant.
            
        Returns
        -------
        TenantConfig
            The configuration for the requested tenant.
            
        Raises
        ------
        ValueError
            If no configuration exists for the tenant.
        """
        if tenant_id not in self.configs:
            logger.error(f"Configuration for tenant {tenant_id} not found.")
            raise ValueError(f"No configuration found for tenant_id: {tenant_id}")
        return self.configs[tenant_id]
