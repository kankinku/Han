from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from pybo.models import Question

# Create your views here.
def index(request):
    #return HttpResponse("<a href='/pybo/1'>안녕하세요 pybo에 오신 것을 환영합니다.</a>")
    question_list =  Question.objects.all()
    context = {'question_list': question_list}
    return render(request, 'pybo/question_list.html', context)

def detail(request, question_id):
    try:
        question = Question.objects.get(id=question_id)
    except:
        question = get_object_or_404(Question, pk=question_id)
    context = {'question': question}
    return render(request, 'pybo/question_detail.html', context)

def answer_create(request, question_id):
    print(request.POST.get('content'))
    print(question_id)
    return redirect('detail', question_id=question_id)