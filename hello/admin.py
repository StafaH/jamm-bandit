from django.contrib import admin

# Register your models here.
from .models import Profile, Log, Arm

admin.site.register(Profile)
admin.site.register(Log)
admin.site.register(Arm)
