from django.contrib import admin

# Register your models here.
from .models import Profile, Log, Arm, DuelRecord, Counter

admin.site.register(Profile)
admin.site.register(Log)
admin.site.register(Arm)
admin.site.register(DuelRecord)
admin.site.register(Counter)
