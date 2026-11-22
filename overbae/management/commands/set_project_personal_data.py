from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from overbae.models import Project, ProjectPersonalDataChange


class Command(BaseCommand):
    help = (
        "Certify whether a project holds personal data. This is the only way to change it: "
        "certified projects are judged by a remote model. Every change is logged."
    )

    def add_arguments(self, parser):
        parser.add_argument("--project-id", required=True)
        parser.add_argument("--slug", help="Optional cross-check: must be the project's slug.")
        parser.add_argument("--personal-data", required=True, choices=["yes", "no"])
        parser.add_argument("--by", required=True, dest="changed_by")
        parser.add_argument("--reason", required=True)

    def handle(self, *args, project_id, slug, personal_data, changed_by, reason, **options):
        if not changed_by.strip() or not reason.strip():
            raise CommandError("--by and --reason must not be blank.")
        value = personal_data == "yes"
        with transaction.atomic():
            try:
                project = Project.objects.select_for_update().filter(pk=project_id).first()
            except (ValidationError, ValueError) as exc:
                raise CommandError(f"{project_id!r} is not a project id.") from exc
            if project is None:
                raise CommandError(f"No project has the id {project_id}.")
            if slug is not None and slug != project.slug:
                raise CommandError(
                    f"Project {project.pk} has the slug {project.slug!r}, not {slug!r}."
                )
            label = f"{project.pk} {project.name!r} ({project.slug})"
            if project.personal_data is value:
                self.stdout.write(f"unchanged: {label} personal_data={personal_data}")
                return
            ProjectPersonalDataChange.objects.create(
                project=project,
                project_id_copy=project.pk,
                project_slug=project.slug,
                old_value=project.personal_data,
                new_value=value,
                changed_by=changed_by.strip(),
                reason=reason.strip(),
            )
            project.personal_data = value
            project.save(update_fields=["personal_data"])
        self.stdout.write(f"changed: {label} personal_data={personal_data}")
