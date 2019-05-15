#!/usr/bin/env python

import sys
import requests

def main():
  serverName = 'localhost'
  serverPort = 5000
  r = requests.get('http://%s:%d/' % (serverName, serverPort))
  if r.status_code != 200:
    print("Error getting index page.")
    print(r.text)
    sys.exit(-1)

  r = requests.post('http://%s:%d/result' % (serverName, serverPort), json = {'A[]': [3.4, 5], 'B[]': [5, 10]})
  if r.status_code != 200:
    print("Error getting results page.")
    print(r.text)
    sys.exit(-1)

  serverName = 'localhost'
  serverPort = 5001
  r = requests.post('http://%s:%d/classify' % (serverName, serverPort), json = {'i': 1, 'A': '3.4', 'B': '5'})
  if r.status_code != 200:
    print("Error getting classification json.")
    print(r.json())
    sys.exit(-1)

if __name__ == '__main__':
  main()

