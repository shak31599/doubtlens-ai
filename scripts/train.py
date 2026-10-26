from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from doubtlens.core import train_all
if __name__=='__main__':
    results=train_all()
    for n in ['baseline','character']:
        print(n, 'accuracy=',round(results[n]['accuracy'],3),'macro F1=',round(results[n]['macro_f1'],3))
    print('Selected via validation:',results['recommended'])
