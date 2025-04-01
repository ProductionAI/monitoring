from memory_profiler import profile
import pandas as pd
import numpy as np
@profile
def my_memory_heavy_function():
    num_rows = 1000000  # Example: 1 million rows
    num_cols = 10       # Example: 10 columns
    data = np.random.rand(num_rows, num_cols) # Generates random numbers between 0 and 1
    df = pd.DataFrame(data)
    df.columns = ['col_' + str(i) for i in range(num_cols)]
    pass

my_memory_heavy_function()
