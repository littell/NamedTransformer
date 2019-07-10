import os
import yaml
import hashlib

def ensure_dirs(path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as fin:
        return yaml.safe_load(fin)
      
def save_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(obj, allow_unicode=True, default_flow_style=False))

def save_txt(output_path, txt):
    ensure_dirs(output_path)
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(txt)

def hash_to_int(obj):
    s = str(obj).encode('utf-8')
    h = hashlib.md5(s).hexdigest()
    return int(h, 16)