"""bmhi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('labwebsite.urls', namespace='labwebsite')),
    path('NIC/', include('NIC.urls', namespace='NIC')),
    path('NIC/MATILDA/', include('MATILDA.urls', namespace='MATILDA')),
    path('epilepsy4d/', include('epilepsy4d.urls')),
    path('kroma/', include('DSapp.urls', namespace='kroma')),
    path('dsai/', include('DSapp.urls', namespace='kroma')),
    path(
        "CDSPD/",
        RedirectView.as_view(
            url="https://cciweb01.case.edu/CDSPD/",
            permanent=False
        ),
        name="projectx-redirect"
    ),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
