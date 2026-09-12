import sys
import traceback
import pandas as pd
from preprocess_vehicle import create_vehicle_windows
from preprocess import find_optimal_shift

try:
    print(create_vehicle_windows)
except Exception as e:
    traceback.print_exc()
