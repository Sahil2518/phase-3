import logging
from typing import Dict, Any, Tuple
from tenant_manager import TenantManager, TenantConfig
from multi_tenant_recommender import MultiTenantRecommender

logger = logging.getLogger(__name__)

class AdminConsole:
    """
    Control plane for enterprise administrators to configure and preview matching policies.
    Enforces strict guardrails to prevent unfair or nonsensical configurations.
    """
    def __init__(self, tenant_manager: TenantManager, recommender: MultiTenantRecommender):
        self.tenant_manager = tenant_manager
        self.recommender = recommender

    def preview_config(self, tenant_id: str, new_config_dict: Dict[str, Any], sample_candidate_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Preview the effect of a configuration change offline before it goes live.
        
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, message, metrics_or_errors)
        """
        # Step 1: Validate config against guardrails
        try:
            proposed_config = TenantConfig(**new_config_dict)
        except Exception as e:
            logger.error(f"Config Guardrail Breach: {str(e)}")
            return False, "Config rejected by guardrails.", {"error": str(e)}

        # Step 2: Backup current live config
        try:
            live_config = self.tenant_manager.get_config(tenant_id)
        except ValueError:
            # If no live config exists, create a default one for comparison
            live_config = TenantConfig()
        
        # Step 3: Run inference with the proposed config temporarily
        # We temporarily set the config in memory, run inference, then revert.
        self.tenant_manager.configs[tenant_id] = proposed_config
        
        try:
            proposed_results = self.recommender.recommend_jobs_for_candidate(tenant_id, sample_candidate_id, k=10)
        except Exception as e:
            # Revert and fail
            self.tenant_manager.configs[tenant_id] = live_config
            return False, f"Inference failed with proposed config: {str(e)}", {"error": str(e)}
            
        # Revert to live config
        self.tenant_manager.configs[tenant_id] = live_config
        
        # Get baseline results for comparison
        try:
            baseline_results = self.recommender.recommend_jobs_for_candidate(tenant_id, sample_candidate_id, k=10)
        except Exception as e:
            baseline_results = []
            
        # Compile preview metrics
        baseline_count = len(baseline_results)
        baseline_avg_score = sum(r['score'] for r in baseline_results) / baseline_count if baseline_count > 0 else 0.0
        
        proposed_count = len(proposed_results)
        proposed_avg_score = sum(r['score'] for r in proposed_results) / proposed_count if proposed_count > 0 else 0.0
        
        metrics = {
            "baseline_matches": baseline_count,
            "baseline_avg_score": round(baseline_avg_score, 4),
            "proposed_matches": proposed_count,
            "proposed_avg_score": round(proposed_avg_score, 4),
            "top_job_id_baseline": baseline_results[0]['id'] if baseline_results else None,
            "top_job_id_proposed": proposed_results[0]['id'] if proposed_results else None,
        }
        
        # Guardrail on preview: Are we returning 0 matches? That might be a bad config.
        if proposed_count == 0 and baseline_count > 0:
            return False, "Preview Warning: Proposed config results in zero matches. Configuration may be too strict.", metrics
            
        return True, "Preview successful.", metrics

    def apply_config(self, tenant_id: str, new_config_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Apply a configuration change live, enforcing guardrails.
        """
        try:
            proposed_config = TenantConfig(**new_config_dict)
            self.tenant_manager.add_or_update_tenant(tenant_id, proposed_config)
            logger.info(f"Tenant {tenant_id} configuration successfully updated.")
            return True, "Configuration successfully applied live."
        except Exception as e:
            logger.error(f"Config update rejected for tenant {tenant_id}: {str(e)}")
            return False, f"Update rejected by guardrails: {str(e)}"
