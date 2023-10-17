import json
import torch
import matplotlib.pyplot as plt

def read_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(data, filepath):
    with open(filepath, "w") as f: 
        json.dump(data, f)

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [i for i in range(torch.cuda.device_count())]

    # print("Found Devices: ", [torch.cuda.get_device_name(i) for i in range(len(device_ids))])

    return device, device_ids

def save_model_state(model, metrics, filepath):    
    torch.save({
        "metrics": metrics,
        "model_weight": model.state_dict(),
    }, filepath)