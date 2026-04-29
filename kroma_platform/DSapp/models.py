from django.db import models
from django.conf import settings

class Article(models.Model):
    pmcid = models.IntegerField(primary_key=True)
    pmid = models.IntegerField()
    title = models.CharField(max_length=200)
    authors = models.CharField(max_length=100)
    first_author = models.TextField(max_length=100)
    journal = models.CharField(max_length=100)
    year = models.IntegerField()
    date = models.DateField(default='2025-01-01')
    doi = models.TextField(max_length=100)
    organism = models.TextField()
    url = models.URLField(max_length=200)
    type = models.TextField(max_length=100)
    ds = models.CharField(max_length=100, null=True, blank=True)
    pdf_path = models.TextField()
    axis = models.CharField(max_length=100)
    abstract = models.TextField(max_length=100)
    fulltext_path = models.TextField(blank=True, default="")
    fulltext_format = models.CharField(max_length=10, blank=True, default="")
    
    class Meta:
        db_table = 'article'
        managed = False
        

class AccessRequest(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    affiliation = models.CharField(max_length=255, blank=True)
    reason = models.TextField(blank=True)

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
    )
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        help_text="Django user created from this request (if any).",
    )

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"


class ChatLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chatlogs",
    )

    # What user asked
    prompt = models.TextField()

    # What the model returned
    response = models.TextField()

    # Other metadata
    user_category = models.CharField(max_length=50, blank=True)
    model_name = models.CharField(max_length=50, blank=True)
    was_success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.created_at} | {self.user} | {self.prompt[:40]}"
