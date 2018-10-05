import re
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.linalg import svd
import nltk
import sys

reload(sys)
sys.setdefaultencoding('utf8')



class word2prob(object):
	"""Map words to their probabilities."""
	def __init__(self, count_fn):
		"""Initialize a word2prob object.

		Args:
			count_fn: word count file name (one word per line) 
		"""
		self.prob = {}
		total = 0.0

		for line in open(count_fn):
			k,v = line.split()
			v = int(v)
			k = k.lower()

			self.prob[k] = v
			total += v

		self.prob = { k : (self.prob[k] / total) for k in self.prob }
		self.min_prob = min(self.prob.values())
		self.count = total

	def __getitem__(self, w):
		return self.prob.get(w.lower(), self.min_prob)

	def __contains__(self, w):
		return w.lower() in self.prob

	def __len__(self):
		return len(self.prob)

	def vocab(self):
		return iter(self.prob.keys())


class word2vec(object):
	"""Map words to their embeddings."""
        def __init__(self, vector_fn):
		"""Initialize a word2vec object.

		Args:
			vector_fn: embedding file name (one word per line)
		"""
		self.vectors = {}
    
		for line in open(vector_fn):
        		line = line.split()

			# skip first line if needed
			if len(line) == 2:
				continue

        		word = line[0]
        		embedding = np.array([float(val) for val in line[1:]])
        		self.vectors[word] = embedding

        def __getitem__(self, w):
		return self.vectors[w]

	def __contains__(self, w):
		return w in self.vectors



class uSIF(object):
	"""Embed sentences using unsupervised smoothed inverse frequency."""
	def __init__(self, vec, prob, n=11, m=3):
		"""Initialize a sent2vec object.

		Variable names (e.g., alpha, a) all carry over from the paper.

		Args:
			vec: word2vec object
			prob: word2prob object
			n: expected random walk length. This is the avg sentence length, which
				should be estimated from a large representative sample. For STS
				tasks, n ~ 11. n should be a positive integer.
			m: number of common discourse vectors (in practice, no more than 5 needed)
		"""
		self.vec = vec
		self.m = m
		self.count=0
		self.size=0

		if not (isinstance(n, int) and n > 0):
			raise TypeError("n should be a positive integer")
	
		vocab_size = float(len(prob))
		threshold = 1 - (1 - 1/vocab_size) ** n
		alpha = len([ w for w in prob.vocab() if prob[w] > threshold ]) / vocab_size
		Z = 0.5 * vocab_size
		self.a = (1 - alpha)/(alpha * Z)
		#print self.a

		self.weight = lambda word: (self.a / (self.a + prob[word])) 

	def _to_vec(self, sentence):
		"""Vectorize a given sentence.
		
		Args:
			sentence: a sentence (string) 
		"""
		# regex for non-punctuation
		not_punc = re.compile('.*[A-Za-z0-9].*')

		# preprocess a given token
		def preprocess(t):
			t = t.lower().strip("';.:()").strip('"')
			t = 'not' if t == "n't" else t
			return t
		
		tokens = map(preprocess, filter(lambda t: not_punc.match(t), nltk.word_tokenize(sentence)))
		tokens = reduce(lambda a,b: a + b, [[]] + map(lambda t: re.split(r'[-]', t), tokens))
		tokens = filter(lambda t: t in self.vec, tokens)

		# if no parseable tokens, return a vector of a's        
		if tokens == []:
			return np.zeros(300) + self.a
		else:
			v_t = np.array(map(lambda (i,t): self.vec[t], enumerate(tokens)))
			v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
			embed1_t = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens)))
			embed1_t = np.mean(embed1_t, axis=0) 

			if (len(v_t) >=12):
				if (len(v_t) % 2 ==0):
					embed2_t_1 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[1:len(tokens)/2])))
					embed2_t_2 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[len(tokens)/2:-1])))
				else: 
					embed2_t_1 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[1:(len(tokens)+1)/2])))
					embed2_t_2 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)+1)/2:-1])))

				if (len(v_t) % 4 ==0):
					embed4_t_1 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[1:len(tokens)/4])))
					embed4_t_2 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[len(tokens)/4:len(tokens)/2])))
					embed4_t_3 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[len(tokens)/2: 3*len(tokens)/4])))
					embed4_t_4 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[ 3*len(tokens)/4:-1])))
				elif (len(v_t) % 4 ==1) : 
					embed4_t_1 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[1:(len(tokens)-1)/4])))
					embed4_t_2 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)-1)/4:(len(tokens)-1)/2])))
					embed4_t_3 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)-1)/2:3*(len(tokens)-1)/4])))
					embed4_t_4 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[3*(len(tokens)-1)/4:-1])))
				elif (len(v_t) % 4 ==2) : 
					embed4_t_1 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[1:(len(tokens)+2)/4])))
					embed4_t_2 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)+2)/4:(len(tokens)+2)/2])))
					embed4_t_3 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)+2)/2: 3*(len(tokens)+2)/4])))
					embed4_t_4 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[3*(len(tokens)+2)/4:-1])))
				else:
					embed4_t_1 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[1:(len(tokens)+1)/4])))
					embed4_t_2 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)+1)/4:(len(tokens)+1)/2])))
					embed4_t_3 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[(len(tokens)+1)/2:3*(len(tokens)+1)/4])))
					embed4_t_4 = np.array(map(lambda (i,t): self.weight(t) * v_t[i,:], enumerate(tokens[3*(len(tokens)+1)/4:-1])))

				embed2_t_1 = np.mean(embed2_t_1, axis=0)
				embed2_t_2 = np.mean(embed2_t_2, axis=0)
				embed2_t = np.maximum(embed2_t_1, embed2_t_2)  #Max Pooling between the two halves

				embed4_t_1 = np.mean(embed4_t_1, axis=0)
				embed4_t_2 = np.mean(embed4_t_2, axis=0)
				embed4_t_3 = np.mean(embed4_t_3, axis=0)
				embed4_t_4 = np.mean(embed4_t_4, axis=0)

				embed4_t = np.maximum(embed4_t_1, embed4_t_2)  #Max Pooling between the two halves
				embed4_t = np.maximum(embed4_t, embed4_t_3)
				embed4_t = np.maximum(embed4_t, embed4_t_4)

				# print np.shape(embed1_t)
				# print np.shape(embed2_t)
				# print np.shape(embed4_t)

				return np.concatenate((embed1_t, embed2_t ,embed4_t), axis=0)
			else:
				return np.concatenate((embed1_t, embed1_t, embed1_t), axis=0)

	def embed(self, sentences):
		"""Embed a list of sentences.

		Args:
			sentences: a list of sentences (strings)
		"""
		vectors = map(self._to_vec, sentences)
		if self.m == 0:
			return vectors

		proj = lambda a, b: a.dot(b.transpose()) * b
		svd = TruncatedSVD(n_components=self.m, random_state=0).fit(vectors)	
	
		# remove the weighted projections on the common discourse vectors
		for i in range(self.m):
			lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
			pc = svd.components_[i]
			vectors = map(lambda v_s: v_s - lambda_i * proj(v_s, pc), vectors)

		return vectors

	
def test_STS(model):
	"""Test the performance on the STS tasks and print out the results.

	Expected results:
		STS2012: 0.683
		STS2013: 0.661
		STS2014: 0.784
		STS2015: 0.790
		SICK2014: 0.735
		STSBenchmark: 0.795

	Args:
		model: a uSIF object
	""" 
	test_dirs = [
		'STS/STS-data/STS2012-gold/',
		#'STS/STS-data/STS2013-gold/',
		'STS/STS-data/STS2014-gold/',
		'STS/STS-data/STS2015-gold/',
		'STS/SICK-data/',
		'STSBenchmark/'
	]

	for td in test_dirs:
		test_fns = filter(lambda fn: '.input.' in fn and fn.endswith('txt'), os.listdir(td))
		scores = []
	
		for fn in test_fns:
			sentences = re.split(r'\t|\n', open(td + fn).read().strip())
			vectors = model.embed(sentences)
			y_hat = [ 1 - cosine(vectors[i], vectors[i+1]) for i in range(0, len(vectors), 2) ]
			y = map(float, open(td + fn.replace('input', 'gs')).read().strip().split('\n'))

			score = pearsonr(y, y_hat)[0]
			scores.append(score)
			print score
			#print fn, "\t", score
		
		print np.mean(scores)
		#print td, np.mean(scores), "\n"


def get_paranmt_usif():
	"""Return a uSIF embedding model that used pre-trained ParaNMT word vectors."""
	prob = word2prob('enwiki_vocab_min200.txt')
	vec = word2vec('vectors/czeng.txt')
	return uSIF(vec, prob)

if __name__== "__main__":
	test_STS(get_paranmt_usif())

