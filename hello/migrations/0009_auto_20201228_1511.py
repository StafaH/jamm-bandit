# Generated by Django 3.1.4 on 2020-12-28 20:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hello', '0008_profile_comments'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='ethnicity',
            field=models.CharField(choices=[('NO', 'None'), ('WH', 'White'), ('HI', 'Hispanic'), ('BL', 'Black'), ('ME', 'Middle Eastern'), ('SA', 'South Asian'), ('SE', 'South-East Asian'), ('EA', 'East Asian'), ('PI', 'Pacific Islander'), ('NA', 'Native American/Indigenous')], default='NO', max_length=2),
        ),
        migrations.AlterField(
            model_name='profile',
            name='gender',
            field=models.CharField(choices=[('N', 'None'), ('M', 'Male'), ('F', 'Female'), ('O', 'Other')], default='NO', max_length=1),
        ),
        migrations.AlterField(
            model_name='profile',
            name='politics',
            field=models.CharField(choices=[('NO', 'None'), ('VL', 'Very Liberal'), ('ML', 'Moderately Liberal'), ('SL', 'Slightly Liberal'), ('NA', 'Neither Liberal or Conservative'), ('VC', 'Very Conservative'), ('MC', 'Moderately Conservative'), ('SC', 'Slightly Conservative')], default='NO', max_length=2),
        ),
    ]