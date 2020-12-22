from django.db import models

# Create your models here.
class arms(models.Model):
    uniformYes = models.IntegerField(0)
    uniformNo = models.IntegerField(0)
    TSYes = models.IntegerField(0)
    TSNo = models.IntegerField(0)
    imgID = models.AutoField(primary_key=True)

class users(models.Model):
    userID = models.IntegerField(0)
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
