import requests
import os
import timeit
import pandas as pd
import json
from utils import Timer
import argparse

def test_html(ip_addr="localhost", batch_size=20):
    proxies = {'http':""}
    url = f'http://{ip_addr}:6357/v1/piidetect'
    urls = pd.read_csv("data/ai_rss.csv")['Permalink']
    urls = urls[:batch_size].to_list()
    payload = {"link_list": json.dumps(urls)}

    with Timer(f"send {len(urls)} link to pii detection endpoint"):
        try:
            resp = requests.post(url=url, data=payload, proxies=proxies) 
            print(resp.text)
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
            print("Request successful!")
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)
            

def test_text(ip_addr="localhost", batch_size=20):
    proxies = {'http':""}
    url = f'http://{ip_addr}:6357/v1/piidetect'
    content = pd.read_csv("data/ai_rss.csv")['Description']
    content = content[:batch_size].to_list()
    payload = {"text_list": json.dumps(content)}

    with Timer(f"send {len(content)} text to pii detection endpoint"):
        try:
            resp = requests.post(url=url, data=payload, proxies=proxies) 
            print(resp.text)
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
            print("Request successful!")
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)         

        
def test_pdf(ip_addr="localhost", batch_size=20):
    proxies = {'http':""}
    url = f'http://{ip_addr}:6357/v1/piidetect'
    dir_path = "data/pdf"
    file_list = os.listdir(dir_path)
    file_list = file_list[:batch_size]
    files = [('files', (f, open(os.path.join(dir_path, f), 'rb'), 'application/pdf')) for f in file_list]
    with Timer(f"send {len(files)} documents to pii detection endpoint"):
        try:
            resp = requests.request('POST', url=url, headers={}, files=files, proxies=proxies) 
            print(resp.text)
            resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
            print("Request successful!")
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_html', action='store_true', help='Test HTML pii detection')
    parser.add_argument('--test_pdf', action='store_true', help='Test PDF pii detection')
    parser.add_argument('--test_text', action='store_true', help='Test Text pii detection')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for testing')
    parser.add_argument('--ip_addr', type=str, default="localhost", help='IP address of the server')
    
    args = parser.parse_args()
    args.ip_addr = "100.83.111.250"
    if args.test_html:
        test_html(ip_addr=args.ip_addr, batch_size=args.batch_size)
    elif args.test_pdf:
        test_pdf(ip_addr=args.ip_addr, batch_size=args.batch_size)
    elif args.test_text:
        test_text(ip_addr=args.ip_addr, batch_size=args.batch_size)
    else:
        print("Please specify the test type")