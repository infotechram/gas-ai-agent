import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("gas_readings.txt", header=None)
data.columns = ["date","weight"]

data["day"] = range(len(data))

X = data[["day"]]
y = data["weight"]

model = LinearRegression()
model.fit(X,y)

current_weight = data["weight"].iloc[-1]

empty_weight = 15

remaining = current_weight - empty_weight

daily_usage = abs(data["weight"].diff().mean())

days_left = int(remaining/daily_usage)

result = f"""
Current Weight: {current_weight}
Average Daily Usage: {round(daily_usage,2)}
Remaining Gas: {round(remaining,2)}
Predicted Days Left: {days_left}
"""

print(result)

with open("result.txt","w") as f:
    f.write(result)