from django.core.management.base import BaseCommand, CommandError
from hello.models import Arm
import glob
import os

class Command(BaseCommand):
    help = 'Creates an arm for every image in /static/images/'

    #def add_arguments(self, parser):
        #parser.add_argument('poll_ids', nargs='+', type=int)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Populating Database...'))

        for name in glob.glob('C:/code/jamm-bandit/hello/static/images/*.bmp'):
            filename = os.path.basename(name)
            # Check if the arm already exists to avoid duplicates
            arm = list(Arm.objects.filter(filename=filename))
            if not arm:
                new_arm = Arm(filename=filename)
                new_arm.save()

                self.stdout.write(self.style.SUCCESS('Successfully created arm for "%s"' % filename))
