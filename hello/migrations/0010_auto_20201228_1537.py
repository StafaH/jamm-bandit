# Generated by Django 3.1.4 on 2020-12-28 20:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('hello', '0009_auto_20201228_1511'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='profile',
            name='age',
        ),
        migrations.RemoveField(
            model_name='profile',
            name='ethnicity',
        ),
        migrations.RemoveField(
            model_name='profile',
            name='gender',
        ),
        migrations.RemoveField(
            model_name='profile',
            name='politics',
        ),
    ]
