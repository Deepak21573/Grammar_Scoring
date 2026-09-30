from pathlib import Path
import re
import tempfile
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
from scipy.sparse import hstack, csr_matrix

BASE=Path(__file__).resolve().parent
MODEL_PATH=BASE/'model.joblib'
app=FastAPI(title='Grammar Scoring Engine',version='1.0')

if not MODEL_PATH.exists():
    raise RuntimeError('model.joblib not found. Run: python train_model.py')

bundle=joblib.load(MODEL_PATH)
model=bundle['model']; vectorizer=bundle['vectorizer']; scaler=bundle['scaler']
whisper=WhisperModel('small',device='cpu',compute_type='int8')
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
    return np.asarray([[len(text),len(words),len(sentences),avg_sent,unique,div(unique,len(words)),div(rep,len(words)),div(filler,len(words)),div(incomplete,len(sentences)),np.mean([len(w) for w in words]) if words else 0,long_sent]],dtype=float)

@app.get('/')
def root(): return {'service':'Grammar Scoring Engine','status':'ok'}

@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    suffix=Path(file.filename or 'audio.wav').suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp:
        tmp.write(await file.read()); path=tmp.name
    try:
        segments,_=whisper.transcribe(path,language='en',beam_size=5,vad_filter=True)
        text=' '.join(s.text.strip() for s in segments)
        Xtext=vectorizer.transform([text])
        Xeng=scaler.transform(features(text))
        X=hstack([Xtext,csr_matrix(Xeng)]).tocsr()
        score=float(np.clip(model.predict(X)[0],0,5))
        return {'grammar_score':round(score,4),'transcript':text}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
