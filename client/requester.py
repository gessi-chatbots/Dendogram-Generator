import requests
import json

base_url = "http://127.0.0.1"
port = "3008"
endpoint = "dendogram/generate"

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
verb_weights = [0.1, 0.25, 0.5, 0.75, 0.9]
obj_weights = [0.9, 0.75, 0.5, 0.25, 0.1]

json_file_path = "body.json"
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

def construct_url(base_url, port, endpoint, params):
    if port:
        full_url = f"{base_url}:{port}/{endpoint}"
    else:
        full_url = f"{base_url}/{endpoint}"
    return f"{full_url}?preprocessing=true&affinity=bert-embedding&metric=cosine&threshold={params['threshold']}&linkage=average&verb-weight={params['verb_weight']}&obj-weight={params['obj_weight']}"

def main():
    for threshold in thresholds:
        for verb_weight, obj_weight in zip(verb_weights, obj_weights):
            params = {
                "threshold": threshold,
                "verb_weight": verb_weight,
                "obj_weight": obj_weight
            }

            url = construct_url(base_url, port, endpoint, params)

            try:
                response = requests.post(url, json=json_data)
                if response.status_code == 200:
                    print(f"Success: {response.status_code} - {url}")
                else:
                    print(f"Error: {response.status_code} - {url}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
