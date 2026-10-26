import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib,pytest,pandas as pd
from doubtlens.core import ROOT,prepare,split,load,predict,retrieve,CATEGORIES

def test_dataset():
    data=prepare(pd.read_csv(ROOT/'data/samples/synthetic_questions.csv'))
    assert set(data.category)==set(CATEGORIES)
    assert data.question_text.nunique()==len(data)

def test_split():
    a,b,c=split(prepare(pd.read_csv(ROOT/'data/samples/synthetic_questions.csv')))
    assert not set(a.question_text)&set(b.question_text)
    assert not set(b.question_text)&set(c.question_text)
    assert not set(a.question_text)&set(c.question_text)

def test_predict():
    models,metrics=load()
    result=predict('Why does Dijkstra fail on negative-weight edges?',models['baseline'])
    assert result['category'] in CATEGORIES
    assert len(result['probabilities'])==4
    assert abs(sum(result['probabilities'].values())-1)<1e-6

def test_invalid():
    models,_=load()
    with pytest.raises(ValueError): predict(' ',models['baseline'])

def test_retrieval():
    index=joblib.load(ROOT/'artifacts/retrieval.joblib')
    assert len(retrieve('How do I implement a hash table with collision resolution?',index))<=5
    exact=index['records'][0]['question_text']
    assert all(row['question']!=exact for row in retrieve(exact,index))

def test_metrics():
    _,data=load();assert 'confusion_matrix' in data['baseline']
