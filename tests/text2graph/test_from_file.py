import os
import subprocess
from urllib.parse import quote
################################################################
#           Test2 
################################################################
# Define the input data : big text file and feed it

curr_dir = os.getcwd()
append_dir = '/tmpdata'
PATH = curr_dir + append_dir
os.system(f"mkdir -p {PATH}")
os.system(f"wget -P {PATH} 'https://gist.githubusercontent.com/wey-gu/75d49362d011a0f0354d39e396404ba2/raw/0844351171751ebb1ce54ea62232bf5e59445bb7/paul_graham_essay.txt'")
text = open(f'{PATH}/paul_graham_essay.txt').read()

encoded_data2 = quote(text[:10000]) # Limiting 10000 to stay within curl post limits

curl_text_cmd2 = ( 
    f"curl -X POST 'http://localhost:8090/v1/text2graph?input_text={encoded_data2}' "
    "-H 'accept: application/json' -d ''"
)

# Execute the curl command
print(f"================ TEST 2 : From text  ======================")
print(f"Input text : {text[:1000]}")
subprocess.run(curl_text_cmd2, shell=True)
