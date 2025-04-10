from django.contrib import admin

from pybo.models import Answer, Question

# Register your models here.
class QustionAdmin(admin.ModelAdmin):
    search_fields = ['subject']
    
class AnswerAdmin(admin.ModelAdmin):
    search_fields=['content']

admin.site.register(Question,QustionAdmin)
admin.site.register(Answer,AnswerAdmin)