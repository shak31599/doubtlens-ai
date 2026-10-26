"""Optional pretrained sentence-embedding classifier and semantic index.

Run explicitly when internet access and sentence-transformers are available.
No fallback is misrepresented as a pretrained transformer.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from .core import ROOT, measure, SEED

MODEL_ID='sentence-transformers/all-MiniLM-L6-v2'

def train_embeddings(root=ROOT):
    from sentence_transformers import SentenceTransformer
    root=Path(root)
    model=SentenceTransformer(MODEL_ID,device='cpu')
    parts={n:pd.read_csv(root/f'data/processed/{n}.csv') for n in ('train','validation','test')}
    vectors={name:model.encode(df.question_text.tolist(),normalize_embeddings=True,batch_size=32,show_progress_bar=False) for name,df in parts.items()}
    best=None
    for c in [.1,1,10]:
        candidate=LogisticRegression(C=c,max_iter=1000,random_state=SEED,class_weight='balanced').fit(vectors['train'],parts['train'].category)
        score=measure_embedding(candidate,parts['validation'],vectors['validation'])['macro_f1']
        if best is None or score>best[0]:best=(score,c,candidate)
    result=measure_embedding(best[2],parts['test'],vectors['test'])
    result['validation_macro_f1']=best[0];result['regularization_C']=best[1];result['model_id']=MODEL_ID
    joblib.dump(best[2],root/'artifacts/embedding_classifier.joblib')
    np.save(root/'artifacts/sentence_embeddings.npy',vectors['train'])
    (root/'artifacts/embedding_metrics.json').write_text(json.dumps(result,indent=2))
    return result

def measure_embedding(classifier,frame,vectors):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    begin=time.perf_counter();pred=classifier.predict(vectors);latency=(time.perf_counter()-begin)*1000/len(frame)
    return {'accuracy':float(accuracy_score(frame.category,pred)),'macro_f1':float(f1_score(frame.category,pred,average='macro')),
            'weighted_f1':float(f1_score(frame.category,pred,average='weighted')),'inference_ms_per_question_classifier_only':latency,
            'confusion_matrix':confusion_matrix(frame.category,pred,labels=classifier.classes_).tolist()}

def semantic_search(question,model,train_vectors,training_records,top_k=5,threshold=.25):
    vector=model.encode([question],normalize_embeddings=True)
    sims=cosine_similarity(vector,train_vectors)[0]
    return [{**training_records[int(i)],'similarity':float(sims[i])} for i in np.argsort(-sims) if sims[i]>=threshold and training_records[int(i)]['question_text'].casefold()!=question.casefold()][:top_k]
