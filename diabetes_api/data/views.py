from rest_framework import mixins, serializers, status
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework.serializers import ModelSerializer, Serializer
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet, GenericViewSet

from data.models import PatientObservation
from data.require_facades import ModelFacade, ModelConfigurationFacade


class PatientObservationSerializer(ModelSerializer):
    class Meta:
        model = PatientObservation
        fields = [
            'id',
            'name', 'date',
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age',
            'label',
            'is_predicted_by_model', 'is_trained',
        ]


class PatientObservationView(ModelViewSet):
    authentication_classes = []
    permission_classes = []
    lookup_field = 'id'
    lookup_url_kwarg = 'pk'
    queryset = PatientObservation.objects.all()
    serializer_class = PatientObservationSerializer


class PatientRequestForPrediction(
    mixins.UpdateModelMixin,
    GenericViewSet,
):
    authentication_classes = []
    permission_classes = []
    lookup_field = 'id'
    lookup_url_kwarg = 'pk'
    queryset = PatientObservation.objects.all()
    serializer_class = PatientObservationSerializer

    def update(self, request, *args, **kwargs):
        instance: PatientObservation = self.get_object()
        if instance.label is not None:
            raise ValidationError(['این کاربر هم اکنون دارای label می‌باشد.'])
        facade = ModelFacade.get_instance()
        result = facade.predict([instance.as_list(consider_label=False)])
        if result['status'] != 'ok':
            raise ValidationError(['خطا در پردازش یادگیری ماشین!'])
        instance.label, instance.is_predicted_by_model = bool(result['data'][0]), True
        instance.save(update_fields=['label', 'is_predicted_by_model'])
        serializer = self.get_serializer(instance)
        return Response(serializer.data)


class ModelConfigurationSerializer(Serializer):
    epsilon = serializers.FloatField(required=False)
    decay = serializers.FloatField(required=False)
    regularizaton = serializers.FloatField(required=False)

    def validate(self, attrs):
        if not attrs.keys():
            raise ValidationError(['دست کم یک تنظیمات را تغییر دهید!'])
        return attrs

    def create(self, validated_data):
        facade = ModelConfigurationFacade.get_instance()
        facade.update_configs(**dict(validated_data))
        return validated_data


class ModelConfigurationView(APIView):
    authentication_classes = []
    permission_classes = []

    def patch(self, request, *args, **kwargs):
        serializer = ModelConfigurationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
