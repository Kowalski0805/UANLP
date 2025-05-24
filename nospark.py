import cudf
import numpy as np

data = np.random.rand(1000000)
df = cudf.DataFrame({'a': data, 'b': data * 2})
print(df.head())
