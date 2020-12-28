from django.conf import settings
from django.db import models
from django.utils import timezone
# Create your models here.


# Arm table, which collects the cumulative statistics of each image
class Arm(models.Model):
    img_id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=50, default="placeholder.jpg")

    uniform_yes = models.IntegerField(default=0)
    uniform_no = models.IntegerField(default=0)
    ts_yes = models.IntegerField(default=0)
    ts_no = models.IntegerField(default=0)

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.filename}'


# Log table will track the individual decisions that user's as they
# go through the surveys
class Log(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    first_arm = models.IntegerField(default=0)
    second_arm = models.IntegerField(default=0)

    choice = models.IntegerField(default=0)

    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.user.username}-{self.timestamp}'


# User Profile to track the user's meta-deta and demographic information
class Profile(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    comments = models.CharField(max_length=500, default="")

    demo_completed = models.BooleanField(default=False)

    uniform_completed = models.BooleanField(default=False)
    uniform_images_seen = models.IntegerField(default=0)

    ts_completed = models.BooleanField(default=False)
    ts_images_seen = models.IntegerField(default=0)

    code = models.CharField(max_length=50)
    # age = models.IntegerField(default=0)

    # GENDERS = [
    #     ('N', 'None'),
    #     ('M', 'Male'),
    #     ('F', 'Female'),
    #     ('O', 'Other')]

    # gender = models.CharField(max_length=1, choices=GENDERS)

    # ETHNICITIES = [
    #     ('NO', 'None'),
    #     ('WH', 'White'),
    #     ('HI', 'Hispanic'),
    #     ('BL', 'Black'),
    #     ('ME', 'Middle Eastern'),
    #     ('SA', 'South Asian'),
    #     ('SE', 'South-East Asian'),
    #     ('EA', 'East Asian'),
    #     ('PI', 'Pacific Islander'),
    #     ('NA', 'Native American/Indigenous')]
    # ethnicity = models.CharField(max_length=2, choices=ETHNICITIES)

    # POLITICS = [
    #     ('NO', 'None'),
    #     ('VL', 'Very Liberal'),
    #     ('ML', 'Moderately Liberal'),
    #     ('SL', 'Slightly Liberal'),
    #     ('NA', 'Neither Liberal or Conservative'),
    #     ('VC', 'Very Conservative'),
    #     ('MC', 'Moderately Conservative'),
    #     ('SC', 'Slightly Conservative')]
    # politics = models.CharField(max_length=2, choices=POLITICS)

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.user.username}'
