import numpy as np

confidence = [0.2, 0.15, 0.5, 0.05, 0.1]

class_name = np.argmax(confidence)

print(class_name)