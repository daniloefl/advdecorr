# app/web.py

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View

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
  data = request.form['data']
  result = {}
  i = 0
  for line in data.split('\n'):
    content = line.split()
    result[i] = {'pvalue' : 0}
    for k in range(0, len(content)):
      result[i]['input_%d' %k] = content[k]
  return render_template('result.html', result = result)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
