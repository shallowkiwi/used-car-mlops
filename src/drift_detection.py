import numpy as np

def detect_drift(input_data):
    # simple logic: if km_driven is unrealistically high
    if input_data[1] > 300000:
        return "Warning: Possible data drift detected!"
    return "No drift detected."