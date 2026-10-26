import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from doubtlens.embedding import train_embeddings
if __name__=='__main__': print(train_embeddings())
