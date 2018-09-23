import logging
import subprocess
import sys
import datetime

# Setup logging
log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)
fh = logging.FileHandler('log.txt', mode='w')
fh.setLevel(logging.DEBUG)
log.addHandler(fh)

log.info('-----polaris-----')
log.info('Platform:\t'+sys.platform)
v = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                     cwd='/'.join(__file__.split('/')[:-1])+'/',
                     stdout=subprocess.PIPE).communicate()[0]
# print('Version:\t'+v.strip().decode('ascii'))
log.info('Version:\t'+v.strip().decode('ascii'))
log.info('Time:\t\t'+datetime.datetime.now().replace(microsecond=0).isoformat())
log.info('Command:\t'+' '.join(sys.argv))
