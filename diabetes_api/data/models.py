from django.db import models, transaction

from data.require_facades import ModelFacade


class PatientObservation(models.Model):
    name = models.CharField(max_length=128)
    date = models.DateField(auto_now_add=True)

    # Data section
    pregnancies = models.PositiveIntegerField()
    glucose = models.PositiveIntegerField()
    blood_pressure = models.PositiveIntegerField()
    skin_thickness = models.PositiveIntegerField()
    insulin = models.PositiveIntegerField()
    bmi = models.FloatField()
    diabetes_pedigree_function = models.FloatField()
    age = models.PositiveIntegerField()
    label = models.BooleanField(default=None, null=True, blank=True)

    is_predicted_by_model = models.BooleanField(default=False)
    is_trained = models.BooleanField(default=False)

    def as_list(self, consider_label=True):
        base = [
            self.pregnancies,
            self.glucose,
            self.blood_pressure,
            self.skin_thickness,
            self.insulin,
            self.bmi,
            self.diabetes_pedigree_function,
            self.age,
        ]
        if consider_label:
            base += [int(self.label)]
        return base

    @classmethod
    def try_training(cls):
        qs = cls.objects.filter(is_trained=False, label__isnull=False)
        if qs.count() < 3:
            return
        data, ids = [], []
        for obs in qs:
            data.append(obs.as_list())
            ids.append(obs.id)
        facade = ModelFacade.get_instance()
        assert facade.train_batch(data)['status'] == 'ok'
        cls.objects.filter(id__in=ids).update(is_trained=True)

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        is_create = not bool(self.pk)
        with transaction.atomic():
            super().save()
            if is_create:
                self.try_training()
