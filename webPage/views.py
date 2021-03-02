from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

from joblib import load
import pandas as pd

passiveAggressiveLinearModel = load('./models/fakeNewsModel.pkl')
IDFVectorizerModel = load('./models/IDFVectorizer.pkl')

def index(request):
    return render(request,'index.html')

def predict(request):
    if request.method == 'POST':
        temp={}
        temp['text']=request.POST.get('news_content')
        testData = pd.DataFrame({'text':temp}).transpose()
        tfidf_test1=IDFVectorizerModel.transform(testData.iloc[0])
        status = passiveAggressiveLinearModel.predict(tfidf_test1)[0]
        context={
            'status':status
        }
    
    return render(request,'index.html',context)
