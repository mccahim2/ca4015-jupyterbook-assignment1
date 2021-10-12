# Data preparation for clustering

```python
import pandas as pd
import numpy as np
```

Importing cleaned data 

```python
all_data = pd.read_csv('data/all_data.csv')
```

```python
all_data.head()
```

:::{note}
(get_dummies)[https://www.geeksforgeeks.org/python-pandas-get_dummies-method/] is a data manipulation function used to convert categorical data into indicator variables
:::

```python
all_data = pd.get_dummies(all_data, columns=["Study_Type"])
```

```python
all_data
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
