import pandas as pd
import numpy as np

sm_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\S-M.csv"
df_sm = pd.read_csv(sm_path, encoding="latin-1")
df_sm.columns = [c.strip() for c in df_sm.columns]

print("Columns:", df_sm.columns.tolist())
print("Ax mean:", df_sm['Ax'].mean(), "std:", df_sm['Ax'].std())
print("Ay mean:", df_sm['Ay'].mean(), "std:", df_sm['Ay'].std())
print("Az mean:", df_sm['Az'].mean(), "std:", df_sm['Az'].std())
print("GravityX mean:", df_sm['GravityX'].mean(), "std:", df_sm['GravityX'].std())
print("GravityY mean:", df_sm['GravityY'].mean(), "std:", df_sm['GravityY'].std())
print("GravityZ mean:", df_sm['GravityZ'].mean(), "std:", df_sm['GravityZ'].std())
