from django.shortcuts import render
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def myapp(request):
    if os.path.exists('model.pkl')==False:
        df = pd.read_csv('lung_cancer_examples.csv')
        df = df.drop('Name', axis=1)
        df = df.drop('Surname', axis=1)
        x = df.drop('Result', axis=1)
        y = df['Result']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        with open('model.pkl', 'wb') as file:
            pickle.dump(clf, file)
    else:
        pass
    return render(request,'index.html')

def pred(request):
    if request.method == 'POST':
        result = 'Prediction Result'
        data = {'Age':[], 'Smokes':[], 'AreaQ':[], 'Alkhol':[]}
        age = data['Age'].append(int(request.POST.get('Age')))
        smoke = data['Smokes'].append(int(request.POST.get('Smokes')))
        area = data['AreaQ'].append(int(request.POST.get('AreaQ')))
        alc = data['Alkhol'].append(int(request.POST.get('Alkhol')))
        with open('model.pkl', 'rb') as file:
            clf = pickle.load(file)
        data = pd.DataFrame(data)
        mod_pred = clf.predict(data)
        if mod_pred == [0]:
            result = 'Patient is save from lung cancer'
        else:
            result = 'Patient will die soon, enjoy'
    return render(request, 'index.html', {'result':result})