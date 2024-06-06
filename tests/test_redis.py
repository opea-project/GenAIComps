import requests
import os
import timeit

class Timer:
    level = 0
    viewer = None
    def __init__(self, name):
        self.name = name
        if Timer.viewer:
            Timer.viewer.display(f"{name} started ...")
        else:
            print(f"{name} started ...")

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if Timer.viewer:
            Timer.viewer.display(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        else:
            print(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
            
proxies = {'http':""}
url = 'http://localhost:6007/v1/dataprep'
dir_path = "data/single"
#dir_path = "data/10files"
#dir_path = "data/pdf"
file_list = os.listdir(dir_path)
file_list = file_list[:1000]
#print(file_list)
files = [('files', (f, open(os.path.join(dir_path, f), 'rb'), 'application/pdf')) for f in file_list]
#print(files)
with Timer(f"ingestion {len(files)} documents to redis"):
    try:
        #resp = requests.post(url=url, files=files, proxies=proxies) 
        resp = requests.request('POST', url=url, headers={}, files=files, proxies=proxies) 
        print(resp.text)
        resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
        print("Request successful!")
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)