# app/web.py

from flask import Flask

app = Flask(__name__)

@app.route('/')
def web()
  return 'Test'

if __name == '__main__':
  app.run(debug=True, host='0.0.0.0')
