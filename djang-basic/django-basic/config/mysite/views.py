from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):     # 요청이 들어올때 어떻게 응답하는가 
    return HttpResponse("Hello, World!")
