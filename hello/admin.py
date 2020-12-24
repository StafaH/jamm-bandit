from django.contrib import admin

# Register your models here.
from .models import Arm, User

admin.site.register(Arm)
admin.site.register(User)
