from sentinel import BatteryResult, KnowledgePack, explain_battery, collect_images

result = BatteryResult.model_validate_json(open("examples/BAT-07.json").read())
pack = KnowledgePack.from_path("knowledge/pack.md")
outcome = explain_battery(result, pack, collect_images(result, root="runs"))

print(outcome.report())          # the loop: attempts, findings, token use
print(outcome.explanation.headline)