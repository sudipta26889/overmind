import django.db.models.deletion
from django.db import migrations, models


def approve_certified(apps, schema_editor):
    # personal_data=False was a certification; it becomes an approval. Nothing else changes.
    Project = apps.get_model("overbae", "Project")
    Project.objects.filter(personal_data=False).update(cloud_judging=True)


def restore_personal_data(apps, schema_editor):
    Project = apps.get_model("overbae", "Project")
    Project.objects.filter(cloud_judging=True).update(personal_data=False)


def invert_log(apps, schema_editor):
    # The log held personal_data values; it now holds cloud_judging values, their negation.
    Change = apps.get_model("overbae", "ProjectCloudJudgingChange")
    for change in Change.objects.all().iterator():
        change.old_cloud_judging = not change.old_cloud_judging
        change.new_cloud_judging = not change.new_cloud_judging
        change.save(update_fields=["old_cloud_judging", "new_cloud_judging"])


class Migration(migrations.Migration):
    dependencies = [
        ("overbae", "0059_project_personal_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="cloud_judging",
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(approve_certified, restore_personal_data),
        migrations.RenameModel("ProjectPersonalDataChange", "ProjectCloudJudgingChange"),
        migrations.RenameField("projectcloudjudgingchange", "old_value", "old_cloud_judging"),
        migrations.RenameField("projectcloudjudgingchange", "new_value", "new_cloud_judging"),
        migrations.AlterField(
            model_name="projectcloudjudgingchange",
            name="project",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="cloud_judging_changes",
                to="overbae.project",
            ),
        ),
        migrations.RunPython(invert_log, invert_log),
        migrations.RemoveField(model_name="project", name="personal_data"),
    ]
