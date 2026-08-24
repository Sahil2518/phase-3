import os
import sys
import json
import logging
from multi_tenant_recommender import TenantCatalogManager, MultiTenantRecommender
from tenant_manager import TenantManager
from admin_console import AdminConsole

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task19.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("Task 19: White-Label Configurability & Admin Control Plane")
    print("=" * 60)
    
    tenant_id = "tenant_A_corp"
    
    # 1. Initialize components
    print("\n[1] Initializing Tenant Manager and Catalog...")
    catalog = TenantCatalogManager(random_state=19)
    catalog.generate_tenant_data(tenant_id, num_candidates=100, num_jobs=50)
    
    # Clean state config file
    config_file = "models/tenant_configs_task19.json"
    if os.path.exists(config_file):
        os.remove(config_file)
        
    tenant_manager = TenantManager(config_file=config_file)
    recommender = MultiTenantRecommender(catalog, tenant_manager)
    recommender.compile_tenant(tenant_id)
    
    admin = AdminConsole(tenant_manager, recommender)
    
    from tenant_manager import TenantConfig
    # Generate default config implicitly (using TenantConfig defaults)
    default_config = TenantConfig()
    tenant_manager.add_or_update_tenant(tenant_id, default_config)

    cands = catalog.get_candidates(tenant_id)
    sample_candidate_id = cands.index[0]
    print(f"Sample Candidate: {sample_candidate_id}")
    print(f"  Location: {cands.loc[sample_candidate_id, 'location']}")
    print(f"  Seniority: {cands.loc[sample_candidate_id, 'seniority']}")
    print(f"  Skills: {cands.loc[sample_candidate_id, 'skills']}")

    print("\n[2] Baseline Recommendations (Default Config)")
    baseline_recs = recommender.recommend_jobs_for_candidate(tenant_id, sample_candidate_id, k=3)
    if not baseline_recs:
        print("  (No recommendations)")
    for r in baseline_recs:
        print(f"  Job {r['id']}: Score={r['score']:.4f}")
        
    print("\n[3] Admin attempts an UNFAIR configuration (Guardrail breach)")
    bad_config_1 = {
        "skill_weight": 0.5,
        "location_weight": 0.3,
        "seniority_weight": 0.2,
        "match_threshold": 0.6,
        "protected_attributes_filter": True  # This should trip the guardrail
    }
    success, msg, metrics = admin.preview_config(tenant_id, bad_config_1, sample_candidate_id)
    print(f"  Attempt: Add protected attribute filter")
    print(f"  Result: Success={success}")
    print(f"  Message: {msg}")
    print(f"  Details: {metrics.get('error')}")

    print("\n[4] Admin attempts a NONSENSICAL configuration (Guardrail breach)")
    bad_config_2 = {
        "skill_weight": 0.8,
        "location_weight": 0.8,
        "seniority_weight": 0.8, # Sum > 1.0
        "match_threshold": 0.6
    }
    success, msg, metrics = admin.preview_config(tenant_id, bad_config_2, sample_candidate_id)
    print(f"  Attempt: Set weights that sum to 2.4")
    print(f"  Result: Success={success}")
    print(f"  Message: {msg}")
    print(f"  Details: {metrics.get('error')}")

    print("\n[5] Admin previews a STRICT but valid configuration")
    strict_config = {
        "skill_weight": 0.8,   # High skill emphasis
        "location_weight": 0.1,
        "seniority_weight": 0.1,
        "match_threshold": 0.85 # Very high threshold
    }
    success, msg, metrics = admin.preview_config(tenant_id, strict_config, sample_candidate_id)
    print(f"  Attempt: Set strict matching config")
    print(f"  Result: Success={success}")
    print(f"  Message: {msg}")
    print(f"  Metrics Preview:")
    print(json.dumps(metrics, indent=4))
    
    # If the threshold is too high and results in 0 matches, the preview returns False
    # Let's adjust if needed to ensure we see a successful preview
    if not success and "zero matches" in msg:
        print("  (Adjusting config to be slightly less strict for demo...)")
        strict_config["match_threshold"] = 0.7
        success, msg, metrics = admin.preview_config(tenant_id, strict_config, sample_candidate_id)
        print(f"  Result: Success={success}")
        print(f"  Message: {msg}")
        print(f"  Metrics Preview:")
        print(json.dumps(metrics, indent=4))

    print("\n[6] Admin applies the STRICT configuration live")
    success, apply_msg = admin.apply_config(tenant_id, strict_config)
    print(f"  Apply Result: {success} - {apply_msg}")
    
    print("\n[7] Live Recommendations with STRICT Config")
    new_recs = recommender.recommend_jobs_for_candidate(tenant_id, sample_candidate_id, k=3)
    if not new_recs:
        print("  (No recommendations passed the strict threshold)")
    for r in new_recs:
        print(f"  Job {r['id']}: Score={r['score']:.4f}")
        
    print("\nEnd-to-End Task 19 complete.")
    
if __name__ == "__main__":
    main()
