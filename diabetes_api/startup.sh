python3 manage.py migrate

uwsgi --http :8000 --module diabetes_api.wsgi