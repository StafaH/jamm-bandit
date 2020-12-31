from django.core.management.base import BaseCommand, CommandError
from hello.models import Arm, DuelRecord
from itertools import combinations
import glob
import os

class Command(BaseCommand):
    help = 'Reset all data for the arms except filename'

    #def add_arguments(self, parser):
        #parser.add_argument('poll_ids', nargs='+', type=int)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Resetting arms...'))

        for arm in Arm.objects.all():
            arm.uniform_yes = 0
            arm.uniform_no = 0
            arm.ts_yes = 0
            arm.ts_no = 0
            arm.save()
            self.stdout.write(self.style.SUCCESS('Successfully reset arm for "%s"' % arm.img_id))
        
        self.stdout.write(self.style.SUCCESS('Clearing duel records'))
        DuelRecord.objects.all().delete()
        arm_combinations = combinations(arms, 2)
        for arm1, arm2 in arm_combinations:
            if DuelRecord.objects.filter(first_arm=arm1.img_id, second_arm=arm2.img_id).count() == 0:
                new_duel = DuelRecord(
                    first_arm=arm1.img_id,
                    second_arm=arm2.img_id,
                )
                new_duel.save()
        self.stdout.write(self.style.SUCCESS('Cleared duel records'))



