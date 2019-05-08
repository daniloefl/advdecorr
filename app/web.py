# app/web.py

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_restful import Resource, Api

import requests

app = Flask(__name__)
bootstrap = Bootstrap(app)
nav = Nav()
nav.init_app(app)

@nav.navigation()
def mynavbar():
    return Navbar(
        'Classifier',
        View('Home', 'index'),
    )

@app.route('/')
@app.route('/home')
def index():
  return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
  A = request.form.getlist('A[]')
  B = request.form.getlist('B[]')
  assert len(A) == len(B)
  N = len(A)
  result = {}
  for i in range(0, N):
    result[i] = {}
    result[i]['pvalue'] = requests.put('http://localhost:5001/classify', data={'i': i, 'A': A[i], 'B': B[i]})
    result[i]['A'] = A[i]
    result[i]['B'] = B[i]
  return render_template('result.html', result = result)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
