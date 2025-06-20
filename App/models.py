from django.db import models

class HandImage(models.Model):
    sign_name = models.CharField(max_length=255)
    image = models.ImageField(upload_to='hand_images/')

    def __str__(self):
        return self.sign_name
    
class SentenceToVideo(models.Model):
    sentence = models.CharField(max_length=1000)
    video = models.FileField(upload_to='videos/')

    def __str__(self):
        return self.sentence
    