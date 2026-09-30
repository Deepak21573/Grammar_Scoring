from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

BASE = Path(__file__).resolve().parent
TRAIN = BASE / 'train.csv'
MODEL = BASE / 'model.joblib'

FILLERS={'um','uh','erm','hmm','like','actually','basically','sort of','kind of','i mean'}

def div(a,b): return float(a)/float(b) if b else 0.0

def features(text):
    text=str(text or '')
    sentences=[s.strip() for s in re.split(r'[.!?]+',text) if s.strip()]
    words=re.findall(r"[A-Za-z]+",text.lower())
    unique=len(set(words))
    rep=sum(words[i]==words[i-1] for i in range(1,len(words)))
    filler=sum(len(re.findall(r"\\b"+re.escape(x)+r"\\b",text.lower())) for x in FILLERS)
    avg_sent=div(len(words),len(sentences))
    incomplete=sum(1 for s in sentences if len(re.findall(r"[A-Za-z]+",s))>=3 and not re.search(r"\\b(is|am|are|was|were|be|been|being|have|has|had|do|does|did|go|went|make|made|explain|explained|work|worked|discuss|discussed|complete|completed|review|reviewed|plan|planned)\\b",s.lower()))
    long_sent=max([len(re.findall(r"[A-Za-z]+",s)) for s in sentences] or [0])
    return [len(text),len(words),len(sentences),avg_sent,unique,div(unique,len(words)),div(rep,len(words)),div(filler,len(words)),div(incomplete,len(sentences)),np.mean([len(w) for w in words]) if words else 0,long_sent]

df=pd.read_csv(TRAIN)
text_col='transcript' if 'transcript' in df.columns else None
if not text_col:
    raise RuntimeError('Demo training data contains transcript. For the real competition, run the notebook transcription step first.')

texts=df[text_col].fillna('').tolist()
y=df['grammar_score'].astype(float).values
X=np.asarray([features(x) for x in texts],dtype=float)

vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=10000,sublinear_tf=True)
Xt=vec.fit_transform(texts)

# Dense engineered features are scaled separately, then concatenated with sparse TF-IDF.
from scipy.sparse import hstack, csr_matrix
scaler=StandardScaler()
Xs=scaler.fit_transform(X)
Xall=hstack([Xt,csr_matrix(Xs)]).tocsr()

model=Ridge(alpha=10.0)
kf=KFold(n_splits=5,shuffle=True,random_state=42)
oof=cross_val_predict(model,Xall,y,cv=kf)
pred=np.clip(oof,0,5)
print(f'CV RMSE: {mean_squared_error(y,pred)**0.5:.4f}')
print(f'CV Pearson: {pearsonr(y,pred)[0]:.4f}')

model.fit(Xall,y)
train_pred=np.clip(model.predict(Xall),0,5)
print(f'TRAINING RMSE (required metric): {mean_squared_error(y,train_pred)**0.5:.4f}')

bundle={'model':model,'vectorizer':vec,'scaler':scaler,'feature_fn_version':1}
joblib.dump(bundle,MODEL)
print('Saved',MODEL)
