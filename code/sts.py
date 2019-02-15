import os
import re
import argparse
from nltk.corpus import stopwords
from nltk import stem
from sklearn.feature_extraction.text import CountVectorizer
import sys
from scipy import spatial
import logging
from gensim.models import word2vec
from itertools import izip
import math
from bs4 import BeautifulSoup
import collections
import numpy as np
from pprint import pprint
import pickle 
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from operator import itemgetter
from scipy.stats.stats import pearsonr 
import cPickle
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
import random
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class sts(object):
	"""
	Implements a regression model to predict semantic similarity scores
	"""
	def __init__(self):
		print 'initializing class'
		self.features= []
		self.Ufeatures =[]
		self.test_features = []
		self.vectorizer = None
		self.preprocessed = []
		self.halfprocessed =[]
		self.model = None
		with open('../model/generalw2v.pickle', "rb") as f:
			self.generalw2v =pickle.load(f)
		 
	def generalw2v(self):
		sentences = word2vec.Text8Corpus('../../downloads/text8')
		model = word2vec.Word2Vec(sentences, size=200, workers =4)
		with open('../model/generalw2v.pickle', "wb") as f:
			pickle.dump(model, f)


	def icf(self):
		stemmer = stem.PorterStemmer()
		sys.stderr.write('getting words for icf dict...\n')
		doc = '../data/all3'
		words = [stemmer.stem(w) for line in open(doc) for w in re.sub(r'\W',' ',line).split(' ')]

		term_count = collections.Counter(words)
		total_count = len(words)
		scale = math.log(total_count)
		sys.stderr.write('calculating icf scores...\n')
		for w, cnt in term_count.iteritems():
			icfdictionary[w] = math.log(total_count / (cnt + 1)) / scale
		pprint(icfdictionary)
		with open('../model/icfdict.pickle', "wb") as f:
			pickle.dump(icfdictionary, f)
			f.close()
	def fit(self):
		#train
		#gridsearch
		pass
	def preprocess(self, file):
		ignored_words = stopwords.words('english')
		stemmer = stem.PorterStemmer()
		sys.stderr.write('preprocessing...\n')
		for line in open(file):
			fields = line.split('\t')
			if len(fields) == 5:
				s1, s2, score, domain, dir = fields
			else:
				s1,s2 = fields
			pattern = re.compile(r'\W')
			s1, s2 = re.sub(pattern,' ',s1), re.sub(pattern,' ',s2)
			s1 = ' '.join([word.lower() for word in s1.split() ])
			s2 = ' '.join([word.lower() for word in s2.split() ])
			self.halfprocessed.append([s1,s2])

			s1 = ' '.join([stemmer.stem(word) for word in s1.split() if word not in ignored_words])
			s2 = ' '.join([stemmer.stem(word) for word in s2.split() if word not in ignored_words])
			self.preprocessed.append([s1,s2])


	def get_features(self, file, save=True ):
		self.preprocess(file)
		print len(self.preprocessed)
		for x in range(len(self.preprocessed)):
			self.features.append([])
		
		self.pos_tag()
		self.w2v()
		self.basic_and_weighted_cosine()
		
		
		if save ==True:
			name = file.split('/')[-1].split('.tx')[0]
			with open('../model/features%s.pickle' %(name), "wb") as f:
				pickle.dump(self.features, f)
				
		
	def load_features(self):
		with open('../model/featurestrain.pickle', "rb") as f:
			self.features =	cPickle.load(f)
			f.close()
	def load_U_features(self):
		with open('../model/featuresunlabelled.pickle', "rb") as f:
			self.Ufeatures =	cPickle.load(f)
			f.close()
	def load_test_features(self, domain):
		with open('../model/featuresSTS.input.%s.pickle' %(domain), "rb") as f:
			self.test_features =	cPickle.load(f)
			f.close()

	def basic_and_weighted_cosine(self,train= True):
		self.vectorizer = CountVectorizer(binary = True)
		sys.stderr.write('creating binary vectors...\n')
		flattened = sum(self.preprocessed, [])
		features = self.vectorizer.fit_transform(flattened)
		feattoarray = features.toarray()

		#get icf score for each vocab item
		with open('../model/icfdict.pickle', "rb") as f:
			self.icfdictionary =pickle.load(f)
		vocabicf = []
		for key, value in self.vectorizer.vocabulary_.iteritems():
			if key in self.icfdictionary:
				vocabicf.append(self.icfdictionary[key])
			else: vocabicf.append(0.97)


		paired = zip(feattoarray[::2], feattoarray[1::2])
		usezero = np.zeros_like(paired[0][0], dtype=np.float)
		for i,pair in enumerate(paired):
			print 'getting basic and weighted cosines... ', i, ' of ', len(paired)
			a,b = pair[0],pair[1]
			usezero2 = usezero.copy()
			usezero3 = usezero.copy()
			cosine = 1- spatial.distance.cosine(a, b)
			#replace with icf scores
			indicesofa = np.nonzero(a)
			indicesofb = np.nonzero(b)
			icfeda =[vocabicf[x] for x in indicesofa[0]]
			icfedb =[vocabicf[x] for x in indicesofb[0]]
			
			yo = np.put(usezero2, list(indicesofa[0]), icfeda)
			yo2 = np.put(usezero3, list(indicesofb[0]), icfedb)

			wcosine = 1- spatial.distance.cosine(usezero2, usezero3)
			print wcosine
			print self.preprocessed[i]
			#print cosine
			if np.isnan(cosine):
				#print 'nan caught ', cosine
				#print 'nan caught ', cosine
				self.features[i].append(0.1)
				self.features[i].append(0.1)
			else:
				#print 'apparently not nan ', cosine
				self.features[i].append(cosine)
				self.features[i].append(wcosine)

	def w2v(self):
		'''
		Inspect actual cosines being returned here and in cosine function...
		inspect actual centroids being calculated
		'''
		model = self.generalw2v
		for i,pair in enumerate(self.halfprocessed):
			print 'getting word vector cosines... ', i, ' of ', len(self.halfprocessed)
			s1,s2 = pair[0], pair[1]
			# print 'sentence and simialrity:'
			# print s1.split(), s2.split()
			s1 = [model[word] for word in s1.split() if word in model.vocab]
			s2 = [model[word] for word in s2.split() if word in model.vocab]
			
				
			if len(s1) ==1:
				centroid1 = np.reshape(np.asarray(s1),(200,))
			else:
				centroid1 = np.mean(s1, axis = 0, dtype=np.float64)
			if len(s2) ==1:
				centroid2 = np.reshape(np.asarray(s2),(200,))
			else:
				centroid2 = np.mean(s2, axis = 0, dtype=np.float64)
			
			print centroid1, centroid2
			print centroid1.shape, centroid2.shape
			cosine_similarity = np.dot(centroid1, centroid2)/(np.linalg.norm(centroid1)* np.linalg.norm(centroid2))
			
			#print cosine_similarity(centroid1, centroid2)
			if len(s1) !=0 and len(s2)!=0:
				self.features[i].append(cosine_similarity)
			else:
				self.features[i].append(0.0)


	def pos_tag(self):
		model = self.generalw2v
		cosines = []
		lex_matches = []
		tags = ['VB', 'NN', 'JJ','RB']
		
		for i,pair in enumerate(self.halfprocessed):
			
			print 'getting pos features... ', i, ' of ', len(self.halfprocessed)
			s1,s2 = pair[0],pair[1]
			cosinesfors1s2 = []
			matchesfors1s2 = []
			a = nltk.word_tokenize(s1)
			b = nltk.word_tokenize(s2)
			posa, posb = nltk.pos_tag(a), nltk.pos_tag(b)
			
			for tag in tags:
				#get number of lexical matches by pos tag
				count = 0
				for x in posa:
					if x in posb and '%s' %(tag) in x[1]:
						count +=1
				
				matchesfors1s2.append(count)
				self.features[i].append(count)
				#get similarity of words with same pos tag
				veca, vecb = [model[x[0]] for x in posa if '%s' %(tag) in x[1] and x[0] in model.vocab], [model[x[0]] for x in posb if '%s' %(tag) in x[1] and x[0] in model.vocab]
				
				if len(veca) != 0 and len(vecb)!= 0:

					centroid1, centroid2 = np.mean(veca,axis = 0, dtype=np.float64), np.mean(vecb,axis = 0, dtype=np.float64)
					cosine_similarity = np.dot(centroid1, centroid2)/(np.linalg.norm(centroid1)* np.linalg.norm(centroid2))
					cosinesfors1s2.append(cosine_similarity)
					self.features[i].append(cosine_similarity)
				else:
					self.features[i].append(0.5)
			lex_matches.append(matchesfors1s2)
			cosines.append(cosinesfors1s2)


	def load_y_gs(self):
		domains = [file.split('.')[-2] for file in os.listdir('../data/by_year_and_domain/test') if 'input' in file]
		gsfiles = [file for file in os.listdir('../data/by_year_and_domain/test') if 'gs' in file]

		#get training data and train first model
		self.load_features()
		x = self.features
		y = [float(y.split('\t')[2].strip()) for y in open('../data/all/train.txt')]

		return domains, gsfiles, x, y

	def gridsearch(self):
		domains, gsfiles, x,y = self.load_y_gs()
		cvalues = np.logspace(-2, 2, 5)

		gammas = np.logspace(-5, 3, 5)
		x = x[::5]
		y = y[::5]
		print gammas, cvalues
		scaler = StandardScaler()
		x = scaler.fit_transform(x)
		print 'data length', len(x), len(y)
		kr = grid_search.GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 10)}, n_jobs= 1, verbose = 10)
		print kr, 'fitting kr'
		kr.fit(x,y)
		print 'best score:  ',kr.best_score_
		print 'best estimator :  ', kr.best_estimator_
		print 'best parameter: ', kr.best_params_

		svr = grid_search.GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e4],
                               "gamma": np.logspace(-2, 2, 10)},n_jobs= 2, verbose = 10)
		print svr
		print 'fitting svr'
		svr.fit(x,y)
		print 'best score:  ',svr.best_score_
		print 'best estimator :  ',svr.best_estimator_
		print 'best parameter: ', kr.best_params_

		

		dtr =  DecisionTreeRegressor()
		clf2 = grid_search.GridSearchCV(estimator=dtr,param_grid={"max_depth":[2,3,4,5,6]}, cv=5, n_jobs= 2, verbose = 10)
		clf2.fit(x,y)
		print 'best score:  ',clf2.best_score_
		print 'best estimator :  ', clf2.best_estimator_
		print 'best parameter: ', clf2.best_params_


	def basictrain(self):
		domains, gsfiles, x,y = self.load_y_gs()
		clf = SVR(C= 10, gamma = 0.01)
		#fit the training data for RBf
		clf.fit(x, y) 
		for domain in domains:
			self.test_features = []
			self.load_test_features(domain)
			
			predictions = clf.predict(self.test_features)

			with open('../data/by_year_and_domain/test/basictrain_%s_output' %(domain), 'a') as f:
				for pred in predictions:
					f.write('%s\n' %(pred))
			scores = self.evaluate('../data/by_year_and_domain/test/basictrain_%s_output'%(domain), '../data/by_year_and_domain/test/STS.gs.%s.txt' %(domain))	
			print scores, domain

	def selftrain(self):
		domains, gsfiles, x,y = self.load_y_gs()
		clf = SVR(C= 10, gamma = 0.01)
		#fit the training data for RBf
		clf.fit(x, y) 

		#load features for unlabelled data, predict on sample and retrain
		self.load_U_features()

		random.shuffle(self.Ufeatures)
		xU = self.Ufeatures
		for pairs in range(0,len(xU),10):
			if pairs+10 < len(xU):
				print xU[pairs:pairs+10]
				for i,array in enumerate(xU[pairs:pairs+10]):
						for v in array:
							if np.isnan(v):
								print 'before predict!!   ',v
								print array
								print i 
				newy = list(clf.predict(xU[pairs:pairs+10]))
				print len(x), len(xU[pairs:pairs+10])
				print len(y), len(newy)
				print type(x), type(xU[pairs:pairs+10])
				print type(y), type(newy)
				x += xU[pairs:pairs+10]
				y += newy
				clf.fit(x,y)
				for domain in domains:
					self.test_features = []
					self.load_test_features(domain)
					
					predictions = clf.predict(self.test_features)

					#delete content of output first because appending multiple times
					with open('../data/by_year_and_domain/test/%s_output' %(domain), "w"):
						pass
					with open('../data/by_year_and_domain/test/%s_output' %(domain), 'a') as f:
						for pred in predictions:
							f.write('%s\n' %(pred))
					scores = self.evaluate('../data/by_year_and_domain/test/%s_output'%(domain), '../data/by_year_and_domain/test/STS.gs.%s.txt' %(domain))	
					print scores, domain, pairs, pairs+10
					with open('../model/selftrainscores.txt', 'a') as f2:
						f2.write('%s, %s, %s\n' %(scores, domain, pairs))

	def tritrain(self):
		domains, gsfiles, x,y = self.load_y_gs()
		x =[[float(feature) for feature in featurelist] for featurelist in x ]
		self.load_U_features()
		
		random.shuffle(self.Ufeatures)
		xU = self.Ufeatures
		xU = [[float(feature) for feature in featurelist] for featurelist in xU]
		print 'training on labeled data'
		clf1  = SVR(C=10,gamma= 0.01, epsilon=0.1)
		clf1.fit(x, y)
		
		clf2 = DecisionTreeRegressor(max_depth=5)
		clf2.fit(x,y)
		clf3 = KernelRidge(alpha = 0.01, gamma= 0.01)
		clf3.fit(x,y)
		allthree = [clf1, clf2, clf3]
		for iteration in range(30):
			for model in allthree:
				x =[[float(feature) for feature in featurelist] for featurelist in self.features ]
				y = [float(y.split('\t')[2].strip()) for y in open('../data/all/train.txt')]
				for pairs in range(0,len(xU),50):
					if pairs+50 < len(xU):
						selection =  xU[pairs:pairs+50]
						#print selection
						useothers = [other for other in allthree if other != model]
						newy0 = useothers[0].predict(selection)
						newy1 = useothers[1].predict(selection)
						newy = []
						newx = []
						for i in range(len(selection)):
							if abs(newy1[i] - newy0[i]) <= 0.2:
								avg = newy0[i] + newy1[i]/ float(2)
								newy.append(avg)
								newx.append(selection[i])
						print len(x), len(y)
						if len(newy) > 0:
							x += newx
							y+= newy
							model.fit(x, y)
							print 'iteration: ' , iteration, 'model: ', model, pairs , 'to ', pairs+1000, 'of ', len(xU)
			for domain in domains:
				self.test_features = []
				self.load_test_features(domain)
				
				predictions = clf1.predict(self.test_features)

				#delete content of output first because appending multiple times
				with open('../data/by_year_and_domain/test/tritrain_%s_output' %(domain), "w"):
					pass
				with open('../data/by_year_and_domain/test/tritrain_%s_output' %(domain), 'a') as f:
					for pred in predictions:
						f.write('%s\n' %(pred))
				scores = self.evaluate('../data/by_year_and_domain/test/tritrain_%s_output'%(domain), '../data/by_year_and_domain/test/STS.gs.%s.txt' %(domain))	
				print scores, domain, pairs, pairs+10


	def baseline(self):
		
		domains, gsfiles, x,y = self.load_y_gs()
		for domain in domains:
			file = '../data/by_year_and_domain/test/STS.input.%s.txt' %(domain)
			self.preprocess(file)
			intersection = []
			with open('../data/by_year_and_domain/test/baseline_%s_output'%(domain) , "wb") as f:
					
				for i,pair in enumerate(self.preprocessed):
					s1,s2 = [word for word in pair[0].split()], [word for word in pair[1].split()]
					intersection = len(set(s1).intersection(s2))
					if intersection != 0:
						percentage1 = intersection/float(len(s1))
						percentage2 = intersection/float(len(s2))
						avg = (percentage2 + percentage1)/2
						
						f.write('%s\n' %(avg * 5))
					else:
						f.write('0\n')
			self.preprocessed =[]
			scores = self.evaluate('../data/by_year_and_domain/test/baseline_%s_output'%(domain), '../data/by_year_and_domain/test/STS.gs.%s.txt' %(domain))	
			print scores, domain

	def evaluate(self, outputfile ,gsfile):
		outputvalues = []
		gsvalues = []

		gs =open(gsfile).readlines()
		outp =open(outputfile).readlines()
		#gs files not complete so...
		for i, line in enumerate(gs):
			if len(line.strip()) != 0:
				gsvalues.append(float(line))
				outputvalues.append(float(outp[i]))
		
		score = pearsonr(outputvalues,gsvalues)
		
		return score


	def build_focused_train_corpus(self, file = None, individual_run = False):
		
		#second training with sub corpus may or may not be run after initial training which builds features and all
		if individual_run:
			self.preprocess(file)
			self.get_features()
		else:
			print ' loading features from pickle file  ...'
			self.load_features()
			
			train_data = self.features
			self.features= []
			self.preprocessed = []
			self.halfprocessed =[]
			self.get_features(file, save= False)

			test_data = self.features
			
			'''
			save the features for test domain?
			'''
			name = file.split('/')[-1]
			with open('../model/%s.pickle' %(name), "wb") as f:
				pickle.dump(test_data, f)	

			fp = open("../data/all/train.txt")
			lines=fp.readlines()	
			focused_training_corpus = []
			for testi,instance in enumerate(test_data):
				cosines= []
				for i, train_instance in enumerate(train_data):
					print 'test instance # ', testi,' of ',len(test_data), ' train instance # ', i , ' of ', len(train_data)
					cosine = np.dot(instance, train_instance)/(np.linalg.norm(instance)* np.linalg.norm(train_instance))
					cosines.append((cosine, i))
				sortedc = sorted(cosines, key=itemgetter(0),reverse = True)
				top5 = sortedc[0:5]
				for rank, (x,index) in enumerate(top5):
					
					focused_training_corpus.append(lines[index])
			focused_training_corpus = list(set(focused_training_corpus))

			domain = name.split('.')[-2]
			with open('../data/focused/%s' %(domain), "wb") as f:
				for x in focused_training_corpus:
					print >> f, x

	def trainfocused(self):
		domains = ['answers-forums', 'answers-students', 'belief']
		clf = SVR(C= 10, gamma = 0.01)
		for domain in domains:

			if 'head' not in domain and 'image' not in domain:
				self.preprocess('../data/focused/%s.txt' %(domain))
				for x in range(len(self.preprocessed)):
					self.features.append([])
				
				self.pos_tag()
				self.w2v()
				self.basic_and_weighted_cosine()
				x = self.features
				y = [float(y.split('\t')[2].strip()) for y in open('../data/focused/%s.txt' %(domain))]
				clf.fit(x,y)
				self.test_features = []
				self.load_test_features(domain)
				
				predictions = clf.predict(self.test_features)

				with open('../data/by_year_and_domain/test/focusedtrain_%s_output' %(domain), 'a') as f:
					for pred in predictions:
						f.write('%s\n' %(pred))
				score = self.evaluate('../data/by_year_and_domain/test/focusedtrain_%s_output'%(domain), '../data/by_year_and_domain/test/STS.gs.%s.txt' %(domain))	
				print score, domain
				with open('../data/by_year_and_domain/test/focusedtrain_scores.txt', 'a+') as f2:
					f2.write('%s %s\n' %(score, domain))
				self.features = []
				self.preprocessed = [] 
				self.halfprocessed =[]
if __name__=="__main__":

	# parse command line options
	parser = argparse.ArgumentParser(description="""Run a sts regression model""")
	parser.add_argument("--fit", help="train regression model", required=False)
	parser.add_argument("--get_features", required = False)
	parser.add_argument("--focused", required=False)
	parser.add_argument("--baseline", required=False,action='store_true')
	parser.add_argument("--icf", required=False,  action='store_true')
	parser.add_argument("--basictrain", required=False,  action='store_true')
	parser.add_argument("--trainfocused", required=False,  action='store_true')

	parser.add_argument("--selftrain", required=False,  action='store_true')
	parser.add_argument("--tritrain", required=False,  action='store_true')
	parser.add_argument("--gridsearch", required=False,  action='store_true')
	args = parser.parse_args()

	# create new model
	sts = sts()

	if args.fit:
	    sts.fit(args.fit)
	if args.icf:
		sts.icf()
	if args.focused:
		sts.build_focused_train_corpus(args.focused)
	if args.baseline:
		sts.baseline()
	if args.basictrain:
		sts.basictrain()
	if args.selftrain:
		sts.selftrain()
	if args.tritrain:
		sts.tritrain()
	if args.gridsearch:
		sts.gridsearch()
	if args.get_features:
		sts.get_features(args.get_features)
	if args.trainfocused:
		sts.trainfocused()
