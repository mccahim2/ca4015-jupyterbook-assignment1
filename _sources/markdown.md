---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: pandoc
      format_version: 2.10.1
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  nbformat: 4
  nbformat_minor: 5
---

::: {.cell .code}
```` {.python}
# Data preparation for clustering

``` python
import pandas as pd
import numpy as np
```

Importing cleaned data

``` python
all_data = pd.read_csv('data/all_data.csv')
```

``` python
all_data.head()
```

(get_dummies)\[\[https://www.geeksforgeeks.org/python-pandas-get_dummies-method/\\\](https://www.geeksforgeeks.org/python-pandas-get_dummies-method/){.uri}\] is a data manipulation function used to convert categorical data into indicator variables

``` python
all_data = pd.get_dummies(all_data, columns=["Study_Type"])
```

``` python
all_data
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```

``` python
```
````
:::