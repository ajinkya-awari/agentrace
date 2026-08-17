import json

from agentrace.nist_report import NISTReportGenerator, NIST_MAPPING


def test_nist_report_maps_exactly_eight_controls():
    report = NISTReportGenerator().generate(mean_rate=35.0, findings={})
    assert set(report["controls"]) == set(NIST_MAPPING)
    json.dumps(report)


def test_deployment_recommendation_thresholds():
    generator = NISTReportGenerator()
    assert generator.deployment_recommendation(19.9) == "deploy"
    assert generator.deployment_recommendation(20.0) == "deploy_with_monitoring"
    assert generator.deployment_recommendation(50.0) == "deploy_with_monitoring"
    assert generator.deployment_recommendation(50.1) == "do_not_deploy"
