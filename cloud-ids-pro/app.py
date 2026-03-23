from flask import Flask, render_template
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)


@app.route("/")
def home():
    # بيانات وهمية للنشاط
    data = {"requests": [10, 12, 11, 13, 100, 9, 10, 110]}
    df = pd.DataFrame(data)

    # نموذج AI
    model = IsolationForest(contamination=0.2)
    model.fit(df)
    df["anomaly"] = model.predict(df)

    # تحويل البيانات لقائمة للعرض
    results = df.to_dict(orient="records")
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)