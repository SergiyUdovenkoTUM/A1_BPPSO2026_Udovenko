#Imports
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as plt
import pm4py as pm
import sklearn as sk
#Store data paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
XES_PATH = DATA_DIR / "BPIC17.xes"
CSV_PATH = DATA_DIR / "BPIC17.csv"

#Main
def main():
    #Import data
    if not XES_PATH.is_file():
        raise FileNotFoundError("XES_PATH does not exist")
    print(f"Reading .xes data from: {XES_PATH}")
    data = pm.read_xes(str(XES_PATH))
    print(f"Data read successfully")

    #Test data
    print(f"First five events:\n" + data.head(5).to_string())


#Run
if __name__ == "__main__":
    main()