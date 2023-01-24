import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import pickle

from flask import Flask, jsonify,request

app = Flask(__name__)

with open('./data/output.pkl','rb') as f:
        rf = pickle.load(f)
    
with open('./data/ss.pkl','rb') as w:
        ss = pickle.load(w)

@app.route('/',methods=['POST'])
def train_data():
    data = request.get_json()
    df = pd.DataFrame.from_dict(data)
    for i in df[['trestbps','fbs','chol','thalach','oldpeak','ca','thal']].columns:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 -q1
        ll = q1 - 1.5 * iqr
        ul = q3 + 1.5 * iqr
        if df[df[i].between(ll,ul)].shape[0] > 0:
            if df[~df[i].between(ll,ul)].shape[0]/df.shape[0]>0.02:
                df[i][df[i]<ll]= ll
                df[i][df[i]>ul]= ul
            else:
                df = df[df[i].between(ll,ul)]

    tran = ss.transform(df)
    df["target"] = rf.predict(tran)
    s = ["age","ca","chol","cp","exang","fbs","oldpeak","restecg","sex","slope","thal","thalach","trestbps","target"]
    df = df[s]
    out = df.to_dict(orient="records")
    return jsonify({
            "result":out,
            "status_code":200,
            "headers": {"content-type": "application/json;charset = utf-8"}
    })

if __name__ == "__main__":
    app.run(debug=True)