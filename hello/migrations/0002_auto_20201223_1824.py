# Generated by Django 3.1.4 on 2020-12-23 23:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hello', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Arm',
            fields=[
                ('uniform_yes', models.IntegerField(verbose_name=0)),
                ('uniform_no', models.IntegerField(verbose_name=0)),
                ('ts_yes', models.IntegerField(verbose_name=0)),
                ('ts_no', models.IntegerField(verbose_name=0)),
                ('img_id', models.AutoField(primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.IntegerField(verbose_name=0)),
                ('name', models.CharField(max_length=50)),
                ('age', models.IntegerField(verbose_name=0)),
                ('gender', models.CharField(choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')], max_length=1)),
                ('ethnicity', models.CharField(choices=[('WH', 'White'), ('HI', 'Hispanic'), ('BL', 'Black'), ('ME', 'Middle Eastern'), ('SA', 'South Asian'), ('SE', 'South-East Asian'), ('EA', 'East Asian'), ('PI', 'Pacific Islander'), ('NA', 'Native American/Indigenous')], max_length=2)),
                ('politics', models.CharField(choices=[('VL', 'Very Liberal'), ('ML', 'Moderately Liberal'), ('SL', 'Slightly Liberal'), ('NA', 'Neither Liberal or Conservative'), ('VC', 'Very Conservative'), ('MC', 'Moderately Conservative'), ('SC', 'Slightly Conservative')], max_length=2)),
            ],
        ),
        migrations.DeleteModel(
            name='Greeting',
        ),
    ]
