from django import forms


class PMCSyncForm(forms.Form):
    search_term = forms.CharField(
        label="Search term",
        max_length=300,
        required=True,
        initial='"Dravet Syndrome" OR "Severe Myoclonic Epilepsy of Infancy" OR SCN1A',
    )
    mindate = forms.DateField(
        label="Start date",
        required=True,
        widget=forms.DateInput(attrs={"type": "date", "min": "1900-01-01"}),
    )
    maxdate = forms.DateField(
        label="End date",
        required=True,
        widget=forms.DateInput(attrs={"type": "date", "min": "1900-01-01"}),
    )
    retmax = forms.IntegerField(label="Max articles", min_value=1, max_value=1000, initial=200)
    overwrite_existing = forms.BooleanField(label="Reprocess existing PMCIDs", required=False)
    download_pdf = forms.BooleanField(label="Try OA PDF download", required=False, initial=True)

    def clean(self):
        cleaned = super().clean()
        mindate = cleaned.get("mindate")
        maxdate = cleaned.get("maxdate")
        if mindate and maxdate and mindate > maxdate:
            self.add_error("mindate", "Start date must be earlier than or equal to end date.")
        return cleaned
