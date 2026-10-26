"""Reproducible question categorization, evaluation, and retrieval."""
from __future__ import annotations
import json, re, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ['DSA', 'Web Development', 'Machine Learning', 'Core Computer Science']
SEED = 42

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s)).strip()

def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'question_id','question_text','category','source','source_url','license'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required-set(frame.columns)}')
    out = frame.copy()
    out['question_text'] = out['question_text'].map(normalize)
    out = out[(out.question_text.str.len() >= 15) & out.category.isin(CATEGORIES)]
    out['key'] = out.question_text.str.casefold()
    return out.drop_duplicates('key').drop(columns='key').reset_index(drop=True)

def split(frame):
    train_val, test = train_test_split(frame, test_size=.2, stratify=frame.category, random_state=SEED)
    train, val = train_test_split(train_val, test_size=.25, stratify=train_val.category, random_state=SEED)
    assert not (set(train.question_text.str.casefold()) & set(test.question_text.str.casefold()))
    return train, val, test

def baseline_model():
    return Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1)),
                     ('clf', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED))])

def alternate_model():
    # Genuine second trainable approach, independent of optional sentence embedding model.
    return Pipeline([('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True, min_df=1)),
                     ('clf', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED))])

def measure(model, test):
    x = test.question_text.tolist(); y = test.category.tolist()
    start = time.perf_counter(); predictions = model.predict(x); elapsed = (time.perf_counter()-start) / len(x) * 1000
    scores = model.predict_proba(x)
    precision, recall, f1, support = precision_recall_fscore_support(y, predictions, labels=CATEGORIES, zero_division=0)
    macro = precision_recall_fscore_support(y, predictions, average='macro', zero_division=0)
    weighted = precision_recall_fscore_support(y, predictions, average='weighted', zero_division=0)
    return {'accuracy':float(accuracy_score(y,predictions)),'macro_precision':float(macro[0]),
            'macro_recall':float(macro[1]),'macro_f1':float(macro[2]),'weighted_f1':float(weighted[2]),
            'inference_ms_per_question':float(elapsed),
            'per_class':{name:{'precision':float(p),'recall':float(r),'f1':float(f),'support':int(s)} for name,p,r,f,s in zip(CATEGORIES,precision,recall,f1,support)},
            'confusion_matrix':confusion_matrix(y,predictions, labels=CATEGORIES).tolist(),
            'classification_report':classification_report(y,predictions,labels=CATEGORIES,zero_division=0),
            'errors':[{'question_text':q,'actual':a,'predicted':p} for q,a,p in zip(x,y,predictions) if a != p],
            'confidence':np.max(scores,axis=1).tolist()}

def train_all(root=ROOT):
    root=Path(root); (root/'artifacts').mkdir(exist_ok=True); (root/'reports').mkdir(exist_ok=True)
    data=prepare(pd.read_csv(root/'data/samples/synthetic_questions.csv'))
    train,val,test=split(data)
    for name,part in [('train',train),('validation',val),('test',test)]: part.to_csv(root/f'data/processed/{name}.csv',index=False)
    results={}; models={}
    for name, factory in [('baseline',baseline_model),('character',alternate_model)]:
        model=factory(); start=time.perf_counter(); model.fit(train.question_text,train.category)
        validation=measure(model,val)
        models[name]=model
        # Model choice is based only on validation, before touching test.
        results[name]={'training_seconds':time.perf_counter()-start,'validation_macro_f1':validation['macro_f1']}
    chosen=max(results,key=lambda k:results[k]['validation_macro_f1'])
    for name,model in models.items():
        results[name].update(measure(model,test));joblib.dump(model,root/f'artifacts/{name}.joblib',compress=3)
    results['recommended']=chosen;results['dataset']={'kind':'original synthetic', 'total':len(data),'train':len(train),'validation':len(val),'test':len(test),'disclaimer':'Synthetic evaluation does not estimate real-world accuracy.'}
    (root/'artifacts/metrics.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame(results[chosen]['errors']).reindex(columns=['question_text','actual','predicted']).to_csv(root/'reports/misclassifications.csv',index=False)
    data.to_csv(root/'data/processed/questions.csv',index=False)
    # Retrieval corpus is deliberately limited to training split, avoiding the held-out test set.
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
    matrix=vectorizer.fit_transform(train.question_text)
    joblib.dump({'vectorizer':vectorizer,'matrix':matrix,'records':train.to_dict('records'),'method':'TF-IDF cosine (offline fallback)'},root/'artifacts/retrieval.joblib',compress=3)
    generate_plots(results,root)
    return results

def generate_plots(results,root):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in ['baseline','character']:
        fig,ax=plt.subplots(figsize=(7,5)); a=np.asarray(results[name]['confusion_matrix']); im=ax.imshow(a,cmap='Blues')
        ax.set_xticks(range(4),CATEGORIES,rotation=35,ha='right');ax.set_yticks(range(4),CATEGORIES)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(a[i,j]),ha='center',va='center')
        ax.set(xlabel='Predicted',ylabel='Actual',title=f'{name.title()} — held-out test (synthetic)')
        fig.tight_layout();fig.savefig(root/f'reports/{name}_confusion.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));names=['baseline','character'];ax.bar(names,[results[n]['macro_f1'] for n in names]);ax.set_ylim(0,1);ax.set_ylabel('Held-out macro F1');fig.tight_layout();fig.savefig(root/'reports/model_comparison.png');plt.close(fig)

def load(root=ROOT):
    root=Path(root)
    metrics=json.loads((root/'artifacts/metrics.json').read_text())
    return {n:joblib.load(root/f'artifacts/{n}.joblib') for n in ['baseline','character']},metrics

def predict(question,model):
    q=normalize(question)
    if len(q)<12: raise ValueError('Enter a technical question of at least 12 characters.')
    probs=model.predict_proba([q])[0]; labels=model.classes_; winner=int(np.argmax(probs))
    return {'category':str(labels[winner]),'confidence':float(probs[winner]),'probabilities':dict(zip(labels.tolist(),map(float,probs))),
            'uncertain':bool(probs[winner]<.55),'note':'Scores are model probabilities, not independently calibrated confidence.'}

def retrieve(question, index, top_k=5, threshold=.05):
    q=normalize(question)
    if len(q)<12: raise ValueError('Enter a longer technical question.')
    query=index['vectorizer'].transform([q]); sim=cosine_similarity(query,index['matrix']).ravel()
    results=[]; seen=set()
    for i in np.argsort(-sim):
        row=index['records'][int(i)]; txt=row['question_text'];key=txt.casefold()
        if sim[i]<threshold or key==q.casefold() or key in seen:continue
        seen.add(key);results.append({'question':txt,'category':row['category'],'source':row['source'],'source_url':row.get('source_url',''),'similarity':round(float(sim[i]),3)})
        if len(results)==top_k:break
    return results

def influential_terms(question,model,count=8):
    tfidf=model.named_steps['tfidf'];classifier=model.named_steps['clf'];x=tfidf.transform([question]);pred=model.predict([question])[0]
    idx=list(classifier.classes_).index(pred);contributions=x.multiply(classifier.coef_[idx]).toarray()[0]
    features=tfidf.get_feature_names_out();inds=np.argsort(-contributions)
    return [(str(features[i]),round(float(contributions[i]),3)) for i in inds if contributions[i]>0][:count]
