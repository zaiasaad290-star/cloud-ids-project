import pandas as pd
from sklearn.ensemble import IsolationForest

# بيانات وهمية (تمثل نشاط المستخدم)
data = {
    "requests": [10, 12, 11, 13, 100, 9, 10, 110],
}

df = pd.DataFrame(data)

# نموذج AI
model = IsolationForest(contamination=0.2)

# تدريب النموذج
model.fit(df)

# التوقع
df["anomaly"] = model.predict(df)

# عرض النتائج
print(df)
