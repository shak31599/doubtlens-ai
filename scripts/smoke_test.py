from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from doubtlens.core import load,predict,retrieve
import joblib
models,metrics=load()
print(predict('Why does a binary search require a sorted array?',models[metrics['recommended']]))
print(retrieve('How do I center a div in CSS?',joblib.load(Path(__file__).resolve().parents[1]/'artifacts/retrieval.joblib'))[:2])
