import torch
from transformers import pipeline
import socket
import json
import pickle
from huggingface_hub import login
login("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")

def start_model_server(host='localhost', port=12370):
    # Load model once
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16,
                     "quantization_config": {"load_in_4bit": True}},
        #device="cuda",
    )
    
    # Create socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    
    print(f"Server listening on {host}:{port}")
    
    while True:
        conn, addr = server_socket.accept()
        try:
            data = b''
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            
            request = pickle.loads(data)
            messages = request['messages']
            print(f"Received request: {messages}")
            max_new_tokens = request.get('max_new_tokens', 256)
            do_sample = request.get('do_sample', False)
            
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            
            conn.sendall(pickle.dumps(response))
        except Exception as e:
            print(f"Error handling request: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    start_model_server()

















