"""Model promotion logic"""
from zenml import step
from typing import Dict
import mlflow
from mlflow.tracking import MlflowClient


@step
def promote_model(
        evaluation_metrics: Dict,
        model_name: str,
        quality_threshold: float = 0.75,
        time_threshold: float = 30.0,
        stage: str = "Production"
) -> Dict:
    """Decide whether to promote model"""

    # Promotion criteria
    quality_ok = evaluation_metrics["avg_quality_score"] >= quality_threshold
    speed_ok = evaluation_metrics["avg_generation_time"] <= time_threshold
    reliability_ok = evaluation_metrics["success_rate"] >= 0.95

    should_promote = quality_ok and speed_ok and reliability_ok

    promotion_result = {
        "promoted": should_promote,
        "model_name": model_name,
        "target_stage": stage,
        "quality_check": quality_ok,
        "speed_check": speed_ok,
        "reliability_check": reliability_ok,
    }

    if should_promote:
        print(f"✓ PROMOTING model to {stage}")
        promotion_result["reason"] = "All checks passed"
    else:
        reasons = []
        if not quality_ok:
            reasons.append(f"Quality {evaluation_metrics['avg_quality_score']:.3f} < {quality_threshold}")
        if not speed_ok:
            reasons.append(f"Speed {evaluation_metrics['avg_generation_time']:.1f}s > {time_threshold}s")
        if not reliability_ok:
            reasons.append(f"Reliability {evaluation_metrics['success_rate']:.2%} < 95%")

        promotion_result["reason"] = "; ".join(reasons)
        print(f"✗ NOT promoting: {promotion_result['reason']}")

    # Log decision
    mlflow.log_dict(promotion_result, "promotion_decision.json")

    return promotion_result