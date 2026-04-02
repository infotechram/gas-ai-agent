import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("gas_readings.txt", names=["date","weight"])

# convert date
data["date"] = pd.to_datetime(data["date"])

# convert to days
data["days"] = (data["date"] - data["date"].min()).dt.days

# need at least 2 readings
if len(data) < 2:
    result="Need at least 2 readings for prediction"
else:

    X = data["days"].values.reshape(-1,1)
    y = data["weight"].values

    model = LinearRegression()
    model.fit(X,y)

    daily_usage = abs(model.coef_[0])

    current_weight = y[-1]
    remaining = current_weight - 0.5

    if daily_usage == 0 or np.isnan(daily_usage):
        result="Not enough usage data yet"
    else:
        days_left = int(remaining/daily_usage)
        result=f"Estimated days remaining: {days_left}"

with open("result.txt","w") as f:
    f.write(result)