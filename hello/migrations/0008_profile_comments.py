# Generated by Django 3.1.4 on 2020-12-28 19:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hello', '0007_auto_20201228_1432'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='comments',
            field=models.CharField(default='', max_length=500),
        ),
    ]
