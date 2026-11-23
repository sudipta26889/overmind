from django.db import migrations


class Migration(migrations.Migration):
    """Ships in the release AFTER 0060 and 0061: until then a worker still on the previous
    release reads ``personal_data``, and dropping the column in the same deploy would break it
    mid-rollout."""

    dependencies = [
        ("overbae", "0061_lexicon_gap_report"),
    ]

    operations = [
        migrations.RemoveField(model_name="project", name="personal_data"),
    ]
