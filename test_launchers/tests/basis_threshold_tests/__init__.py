import sys
from os.path import dirname, join
dir = dirname(dirname(dirname(dirname(__file__))))
sys.path.insert(1, join(dir))
