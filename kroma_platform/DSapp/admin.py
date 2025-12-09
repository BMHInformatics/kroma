from django.contrib import admin
from django.contrib.auth import get_user_model
from .models import AccessRequest
from django.http import HttpResponse
import csv


@admin.register(AccessRequest)
class AccessRequestAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "first_name",
        "last_name",
        "email",
        "affiliation",
        "status",
        "user",
    )
    list_filter = ("status", "created_at", "affiliation")
    search_fields = ("first_name", "last_name", "email", "affiliation", "reason")
    readonly_fields = ("created_at", "user")
    ordering = ("-created_at",)

    actions = ["mark_as_approved", "mark_as_rejected", "create_users_from_requests", "export_as_csv", ]

    def mark_as_approved(self, request, queryset):
        updated = queryset.update(status="approved")
        self.message_user(request, f"{updated} request(s) marked as approved.")
    mark_as_approved.short_description = "Mark selected requests as APPROVED"

    def mark_as_rejected(self, request, queryset):
        updated = queryset.update(status="rejected")
        self.message_user(request, f"{updated} request(s) marked as rejected.")
    mark_as_rejected.short_description = "Mark selected requests as REJECTED"

    def create_users_from_requests(self, request, queryset):
        """
        For each selected AccessRequest:
        - If no user is linked yet, create a Django user.
        - Link it to the request and mark status as 'approved'.
        """
        User = get_user_model()
        created_count = 0
        skipped_existing_user = 0
        skipped_already_linked = 0

        for access_req in queryset:
            # Skip if a user is already linked
            if access_req.user is not None:
                skipped_already_linked += 1
                continue

            # Use email as username
            username = access_req.email

            # If a user with this username already exists, skip
            if User.objects.filter(username=username).exists():
                skipped_existing_user += 1
                continue

            # Create the user with an unusable password
            user = User.objects.create_user(
                username=username,
                email=access_req.email,
                first_name=access_req.first_name,
                last_name=access_req.last_name,
            )
            # Set an unusable password:
            # user.set_unusable_password()
            # user.save()

            access_req.user = user
            access_req.status = "approved"
            access_req.save()

            created_count += 1

        msg = f"Created {created_count} user(s)."
        if skipped_existing_user:
            msg += f" Skipped {skipped_existing_user} request(s) because a user with that username already exists."
        if skipped_already_linked:
            msg += f" Skipped {skipped_already_linked} request(s) already linked to a user."

        self.message_user(request, msg)

    create_users_from_requests.short_description = "Create Django users from selected requests"

    def export_as_csv(self, request, queryset):
        """
        Export selected access requests as a CSV file.
        """
        # Define the response metadata
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="kroma_access_requests.csv"'

        writer = csv.writer(response)

        # Header row
        writer.writerow([
            "created_at",
            "first_name",
            "last_name",
            "email",
            "affiliation",
            "reason",
            "status",
            "user_username",
            "user_email",
        ])

        # Data rows
        for req in queryset:
            writer.writerow([
                req.created_at,
                req.first_name,
                req.last_name,
                req.email,
                req.affiliation,
                req.reason.replace("\n", " ") if req.reason else "",
                req.status,
                req.user.username if req.user else "",
                req.user.email if req.user else "",
            ])

        return response

    export_as_csv.short_description = "Export selected requests as CSV"