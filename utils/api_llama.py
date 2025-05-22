# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re, pdb
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

from transformers import pipeline
import torch
import ssl
import urllib.request
import zipfile
import socket
import pickle


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

def LlamaChatCompletion(model_name, prompt, max_tokens,temperature):

 


    if model_name.lower() == 'Llama-3.3-70b-instruct'.lower():
        HOST = 'localhost'
        PORT = 12370
        request = {
            'messages': [{"role": "user", "content": prompt}],
            'max_new_tokens': max_tokens,
            'do_sample': temperature > 0
        }
        
        # Create socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(pickle.dumps(request))
            s.shutdown(socket.SHUT_WR)
            
            data = b''
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
                
            response = pickle.loads(data)
        
        return response
     
            


