import os

from parameters import ROOT_DIR
from utils import save_txt

files = os.listdir(os.path.join(ROOT_DIR, 'logs'))
logs = []
for f in files:
    if f.endswith('.txt'):
        logs.append(f)

txt = ""
for log in logs:
    save_txt(txt=txt, txt_path=os.path.join(ROOT_DIR, 'logs', log), mode='w')
