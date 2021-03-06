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

    SAMPLING_CHOICES = [("ts", "Double Thompson Sampling"), ("un", "Uniform Sampling")]

    first_arm = models.IntegerField(default=0)
    second_arm = models.IntegerField(default=0)

    choice = models.IntegerField(default=0)

    sampling_type = models.CharField(choices=SAMPLING_CHOICES, max_length=2, default="un")

    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.user.username}-{self.timestamp}'

# Record of wins and losses between arms
class DuelRecord(models.Model):
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['first_arm', 'second_arm'], name='unique_duels')
        ]

    first_arm = models.IntegerField(db_index=True)
    second_arm = models.IntegerField(db_index=True) 

    first_arm_wins = models.IntegerField(default=0)
    second_arm_wins = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.first_arm} vs {self.second_arm}"

# Keeps track of how many pairs of comparisons have been completed overall by all participants
class Counter(models.Model):
    uniform_count = models.IntegerField(default=0)
    ts_count = models.IntegerField(default=0)
    total_count = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.total_count}"

    def save(self, *args, **kwargs):
        if not self.pk and Counter.objects.exists():
            raise ValidationError('Only one counter can exist at once (it keeps track of all trials conducted overall)')
        return super(Counter, self).save(*args, **kwargs)

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
