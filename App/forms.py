from django import forms

class HandSignForm(forms.Form):
    new_sign = forms.CharField(
        max_length=255,
        label='New Sign',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Name of New Sign',
        })
    )