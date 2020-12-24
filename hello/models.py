from django.db import models
# Create your models here.


class Arm(models.Model):
    uniform_yes = models.IntegerField(0)
    uniform_no = models.IntegerField(0)
    ts_yes = models.IntegerField(0)
    ts_no = models.IntegerField(0)
    img_id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=50)

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.filename}'


class User(models.Model):
    user_id = models.IntegerField(0)
    name = models.CharField(max_length=50)
    age = models.IntegerField(0)

    GENDERS = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other')]

    gender = models.CharField(max_length=1, choices=GENDERS)

    ETHNICITIES = [
        ('WH', 'White'),
        ('HI', 'Hispanic'),
        ('BL', 'Black'),
        ('ME', 'Middle Eastern'),
        ('SA', 'South Asian'),
        ('SE', 'South-East Asian'),
        ('EA', 'East Asian'),
        ('PI', 'Pacific Islander'),
        ('NA', 'Native American/Indigenous')]
    ethnicity = models.CharField(max_length=2, choices=ETHNICITIES)

    POLITICS = [
        ('VL', 'Very Liberal'),
        ('ML', 'Moderately Liberal'),
        ('SL', 'Slightly Liberal'),
        ('NA', 'Neither Liberal or Conservative'),
        ('VC', 'Very Conservative'),
        ('MC', 'Moderately Conservative'),
        ('SC', 'Slightly Conservative')]
    politics = models.CharField(max_length=2, choices=POLITICS)

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.name}'
