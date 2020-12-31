# Generated by Django 3.1.4 on 2020-12-25 20:39

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('hello', '0002_auto_20201223_1824'),
    ]

    operations = [
        migrations.CreateModel(
            name='History',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uniform_completed', models.BooleanField(default=False)),
                ('uniform_images_seen', models.IntegerField(default=0)),
                ('ts_completed', models.BooleanField(default=False)),
                ('ts_images_seen', models.IntegerField(default=0)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Log',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_arm', models.IntegerField(default=0)),
                ('second_arm', models.IntegerField(default=0)),
                ('choice', models.IntegerField(default=0)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('age', models.IntegerField(default=0)),
                ('gender', models.CharField(choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')], max_length=1)),
                ('ethnicity', models.CharField(choices=[('WH', 'White'), ('HI', 'Hispanic'), ('BL', 'Black'), ('ME', 'Middle Eastern'), ('SA', 'South Asian'), ('SE', 'South-East Asian'), ('EA', 'East Asian'), ('PI', 'Pacific Islander'), ('NA', 'Native American/Indigenous')], max_length=2)),
                ('politics', models.CharField(choices=[('VL', 'Very Liberal'), ('ML', 'Moderately Liberal'), ('SL', 'Slightly Liberal'), ('NA', 'Neither Liberal or Conservative'), ('VC', 'Very Conservative'), ('MC', 'Moderately Conservative'), ('SC', 'Slightly Conservative')], max_length=2)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.DeleteModel(
            name='User',
        ),
        migrations.AddField(
            model_name='arm',
            name='filename',
            field=models.CharField(default='placeholder.jpg', max_length=50),
        ),
        migrations.AlterField(
            model_name='arm',
            name='ts_no',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='arm',
            name='ts_yes',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='arm',
            name='uniform_no',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='arm',
            name='uniform_yes',
            field=models.IntegerField(default=0),
        ),
    ]