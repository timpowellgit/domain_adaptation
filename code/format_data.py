import os
from itertools import izip


trainfile = open('../data/all/train.txt', 'a')
testfile = open('../data/all/test.txt', 'a')
for dir, subdirs, files in os.walk('../data/by_year_and_domain/train'):
	for f in files:
		if 'input' in f:
			domain = f.split('.')[-2]
			gsfile = 'STS.gs.%s.txt' %(domain)
			year = dir.split('/')[4]
			for line_from_file_1, line_from_file_2 in izip(open(os.path.join(dir,f)), open(os.path.join(dir,gsfile))):
				print >> trainfile, line_from_file_1[:-1], "\t", line_from_file_2[:-1],"\t", domain, "\t", year

