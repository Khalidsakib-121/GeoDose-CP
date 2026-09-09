from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from geodose_stage5b.runner import run
if __name__=='__main__':
    r=run(ROOT)
    print('\nSTAGE5B PRODUCTION GENERATION COMPLETE')
    print(r)
