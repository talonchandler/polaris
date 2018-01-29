import subprocess
cmd = ['git', 'rev-parse', 'HEAD']
cwd = '/'.join(__file__.split('/')[:-1])+'/'
out = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE).communicate()[0]
__version__ = out.strip().decode('ascii')
