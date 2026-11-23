from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from overbae.models import Project, ProjectCloudJudgingChange


class Command(BaseCommand):
    help = (
        "Approve or withdraw masked remote judging for a project. This is the only way to change "
        "it, and every change is logged."
    )

    def add_arguments(self, parser):
        parser.add_argument("--project-id", required=True)
        parser.add_argument("--slug", help="Optional cross-check: must be the project's slug.")
        parser.add_argument("--cloud-judging", required=True, choices=["on", "off"])
        parser.add_argument("--by", required=True, dest="changed_by")
        parser.add_argument("--reason", required=True)

    def handle(self, *args, project_id, slug, cloud_judging, changed_by, reason, **options):
        if not changed_by.strip() or not reason.strip():
            raise CommandError("--by and --reason must not be blank.")
        value = cloud_judging == "on"
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
            if project.cloud_judging is value:
                self.stdout.write(f"unchanged: {label} cloud_judging={cloud_judging}")
                return
            ProjectCloudJudgingChange.objects.create(
                project=project,
                project_id_copy=project.pk,
                project_slug=project.slug,
                old_cloud_judging=project.cloud_judging,
                new_cloud_judging=value,
                changed_by=changed_by.strip(),
                reason=reason.strip(),
            )
            project.cloud_judging = value
            project.save(update_fields=["cloud_judging"])
        self.stdout.write(f"changed: {label} cloud_judging={cloud_judging}")
