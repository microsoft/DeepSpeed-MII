import json
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", type=str)
args = parser.parse_args()

uri = "http://localhost:6789/score"

input_data = {"query": args.query}

headers = {'Content-Type': 'application/json'}
resp = requests.post(uri, json.dumps(input_data), headers=headers)
print(resp.text)
