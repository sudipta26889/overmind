from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from overbae.models import Project, ProjectPersonalDataChange


class Command(BaseCommand):
    help = (
        "Certify whether a project holds personal data. This is the only way to change it: "
        "certified projects are judged by a remote model. Every change is logged."
    )

    def add_arguments(self, parser):
        parser.add_argument("slug")
        parser.add_argument("--personal-data", required=True, choices=["yes", "no"])
        parser.add_argument("--by", required=True, dest="changed_by")
        parser.add_argument("--reason", required=True)

    def handle(self, *args, slug, personal_data, changed_by, reason, **options):
        if not changed_by.strip() or not reason.strip():
            raise CommandError("--by and --reason must not be blank.")
        value = personal_data == "yes"
        with transaction.atomic():
            project = Project.objects.select_for_update().filter(slug=slug).first()
            if project is None:
                raise CommandError(f"No project has the slug {slug!r}.")
            if project.personal_data is value:
                self.stdout.write(f"unchanged: {slug} personal_data={personal_data}")
                return
            ProjectPersonalDataChange.objects.create(
                project=project,
                old_value=project.personal_data,
                new_value=value,
                changed_by=changed_by.strip(),
                reason=reason.strip(),
            )
            Project.objects.filter(pk=project.pk).update(personal_data=value)
        self.stdout.write(f"changed: {slug} personal_data={personal_data}")
