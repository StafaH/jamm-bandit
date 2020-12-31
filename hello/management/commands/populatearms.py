from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from hello.models import Arm, DuelRecord, Counter
from itertools import combinations
import glob
import os

class Command(BaseCommand):
    help = 'Creates an arm for every image in /static/images/'

    #def add_arguments(self, parser):
        #parser.add_argument('poll_ids', nargs='+', type=int)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Populating Database...'))

        filenames = glob.glob(os.path.join(settings.STATIC_ROOT, 'images/*.bmp'))
       
        if not filenames:
            self.stdout.write(self.style.SUCCESS('No images were found'))

        for name in filenames:
            filename = os.path.basename(name)

            # Django adds duplicates of the images that we have to ignore
            if len(filename) > 16: 
                continue
            # Check if the arm already exists to avoid duplicates
            arm = list(Arm.objects.filter(filename=filename))
            if not arm:
                new_arm = Arm(filename=filename)
                new_arm.save()

                self.stdout.write(self.style.SUCCESS('Successfully created arm for "%s"' % filename))
            else:
                self.stdout.write(self.style.SUCCESS('Already exists: "%s"' % filename))
        
        self.stdout.write(self.style.SUCCESS('Creating duel log for arms...'))

        # Initialize duel history between arms
        arms = Arm.objects.all()
        arm_combinations = combinations(arms, 2)
        for arm1, arm2 in arm_combinations:
            if DuelRecord.objects.filter(first_arm=arm1.img_id, second_arm=arm2.img_id).count() == 0:
                new_duel = DuelRecord(
                    first_arm=arm1.img_id,
                    second_arm=arm2.img_id,
                )
                new_duel.save()

        self.stdout.write(self.style.SUCCESS('Successfully created duel log'))

        self.stdout.write(self.style.SUCCESS('Starting counter'))
        if Counter.objects.all().count() > 1:
            Counter.objects.all().delete()
        counter = Counter()
        counter.save()



