from collections import OrderedDict
import argparse
import glob
import json
import os
import shutil
import subprocess
import uuid
from collections import OrderedDict

from joblib import delayed
from joblib import Parallel
import pandas as pd



a = '.\\tmp\\kinetics\\3yaoNwz99xM_000062_000072.mp4'
#b = '\\tmp\\kinetics\\1c5947e9-0e3b-4083-8ff1-fbff555fdbd0.%(ext)s'
print (a.split('.')[0])
print ('%s*' % a.split('.')[0])
#print (glob.glob('.\\tmp\\kinetics\\*'))
print (glob.glob(a[0:-5]+'*'))
#print (glob.glob('%s*' % a.split('.')[0])[0])
#print (b.split('.')[0])
#print (glob.glob('%s*' % b.split('.')[0])[0])
'''
command = ['ffmpeg',
               '-i', '"%s"' % '\\tmp\kinetics\\b983db0c-d0c5-4db4-a839-db39b2f813df.%(ext)s',
               '-ss', str(2),
               '-t', str(10),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % '.\\']
#    command = ' '.join(command)

output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
'''