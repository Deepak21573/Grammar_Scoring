import subprocess,sys,joblib,re
import numpy as np,pandas as pd
from pathlib import Path
from scipy.sparse import hstack,csr_matrix

subprocess.check_call([sys.executable,'train_model.py'])
base=Path(__file__).resolve().parent
bundle=joblib.load(base/'model.joblib'); test=pd.read_csv(base/'test.csv')
F={'um','uh','erm','hmm','like','actually','basically','sort of','kind of','i mean'}
def div(a,b): return float(a)/float(b) if b else 0.0
def feat(text):
 text=str(text or ''); s=[x.strip() for x in re.split(r'[.!?]+',text) if x.strip()]; w=re.findall(r'[A-Za-z]+',text.lower()); u=len(set(w)); rep=sum(w[i]==w[i-1] for i in range(1,len(w))); filler=sum(len(re.findall(r'\\b'+re.escape(x)+r'\\b',text.lower())) for x in F); inc=sum(1 for z in s if len(re.findall(r'[A-Za-z]+',z))>=3 and not re.search(r'\\b(is|am|are|was|were|be|been|being|have|has|had|do|does|did|go|went|make|made|explain|explained|work|worked|discuss|discussed|complete|completed|review|reviewed|plan|planned)\\b',z.lower())); long=max([len(re.findall(r'[A-Za-z]+',z)) for z in s] or [0]); return [len(text),len(w),len(s),div(len(w),len(s)),u,div(u,len(w)),div(rep,len(w)),div(filler,len(w)),div(inc,len(s)),np.mean([len(x) for x in w]) if w else 0,long]
Xtext=bundle['vectorizer'].transform(test.transcript.fillna('')); Xeng=bundle['scaler'].transform(np.asarray([feat(x) for x in test.transcript.fillna('')])); X=hstack([Xtext,csr_matrix(Xeng)]).tocsr(); pred=np.clip(bundle['model'].predict(X),0,5)
pd.DataFrame({'filename':test.filename,'grammar_score':pred}).to_csv(base/'submission.csv',index=False)
print('\\nDone. submission.csv generated with',len(pred),'predictions.')
