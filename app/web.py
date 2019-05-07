# app/web.py

import flask

app = flask.Flask(__name__)

@app.route('/')
@app.route('/home')
def web():
  return flask.render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
  data = flask.request.form['data']
  result = {}
  i = 0
  for line in data.split('\n'):
    content = line.split()
    result[i] = {'pvalue' : 0}
    for k in range(0, len(content)):
      result[i]['input_%d' %k] = content[k]
  return flask.render_template('result.html', result = result)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
