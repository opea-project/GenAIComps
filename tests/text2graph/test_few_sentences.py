import os
import subprocess
from urllib.parse import quote
################################################################
#           Test1 
################################################################
# Define the input data : Few sentences 
text = "I am going to London to visit the queen. After, I will meet the king and have dinner."
encoded_data = quote(text)

curl_text_cmd1 = ( 
    f"curl -X POST 'http://localhost:8090/v1/text2graph?input_text={encoded_data}' "
    "-H 'accept: application/json' -d ''"
)

# Execute the curl command
print(f"================ TEST 1 ======================")
print(f"Input text : {text}")
subprocess.run(curl_text_cmd1, shell=True)
