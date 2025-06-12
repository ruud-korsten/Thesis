from data_quality_tool.evaluation.explain_violation import ExplainViolation

explainer = ExplainViolation(run_name="2025-05-27_14-06-57")
print(explainer.explain_violation(index=359077))
