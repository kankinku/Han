from django.contrib import admin

from pybo.models import Answer, Question

class QuestionAdmin(admin.ModelAdmin):
    list_display = ['subject','create_date']
    def __str__(self):
        return super().__str__()

# Register your models here.
admin.site.register(Question, QuestionAdmin)
admin.site.register(Answer)

# 입력하고 ctrl + . 을 통해서 import 해주면 된다.
