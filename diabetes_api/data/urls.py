from django.urls import path

from data.views import PatientObservationView, PatientRequestForPrediction, ModelConfigurationView

urlpatterns = [
    path('patient/', PatientObservationView.as_view({
        'post': 'create'
    })),
    path('patient/<int:pk>/', PatientObservationView.as_view({
        'get': 'retrieve',
        'patch': 'update',
        'delete': 'delete',
    })),
    path('patient/<int:pk>/predict/', PatientRequestForPrediction.as_view({
        'patch': 'update',
    })),
    path('model/', ModelConfigurationView.as_view()),
]
