from django.db import migrations

MESSAGE = "Ended: external model participants are no longer supported."
TERMINAL = ("completed", "failed")


def end_external_work(apps, schema_editor):
    Plan = apps.get_model("overbae", "NativeEvaluationPlan")
    Run = apps.get_model("overbae", "DecisionPerformanceRun")
    ended = [
        plan.pk
        for plan in Plan.objects.exclude(state__in=TERMINAL)
        if any(p.get("kind") == "external" for p in plan.config.get("participants", []))
    ]
    Plan.objects.filter(pk__in=ended).update(state="failed", error=MESSAGE)
    Run.objects.filter(evaluation_id__in=ended).exclude(state__in=TERMINAL).update(
        state="failed", error=MESSAGE
    )


class Migration(migrations.Migration):
    dependencies = [
        ("overbae", "0056_mcp_mutations_and_eval_revisions"),
    ]

    operations = [
        migrations.RunPython(end_external_work, migrations.RunPython.noop),
    ]
