from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from DSapp import views

app_name = 'DSapp'

urlpatterns = [
    path('', auth_views.LoginView.as_view(
        template_name='DSapp/login.html'
    ), name='login'),
    path('home/', views.index, name='index'),
    path('kg-chat-api/', views.kg_chat_api, name='kg_chat_api'),
    path('logout/',
         auth_views.LogoutView.as_view(next_page='DSapp:login'),
         name='logout'),
    path('request-access/', views.request_access, name='request_access'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
