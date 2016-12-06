import re
import math
import scipy.linalg
import numpy
import operator
import argparse

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy import spatial


# input from commands
parser = argparse.ArgumentParser()
parser.add_argument('-z', dest='z')
parser.add_argument('-k', dest='k')
parser.add_argument('--dir', dest='dir')
parser.add_argument('--doc_in', dest='doc_in')
parser.add_argument('--doc_out', dest='doc_out')
parser.add_argument('--term_in', dest='term_in')
parser.add_argument('--term_out', dest='term_out')
parser.add_argument('--query_in', dest='query_in')
parser.add_argument('--query_out', dest='query_out')
args = parser.parse_args()
z = int(args.z)		# Dimensionality of lower dimensional space
k = int(args.k)		# k similar terms/documents to be returned
directory = args.dir 		# directory of papers
doc_in  = args.doc_in
doc_out = args.doc_out
term_in = args.term_in
term_out = args.term_out
query_in = args.query_in
query_out = args.query_out

numberOfDocuments=5000	# total number of documents/papers given.
documents=[]		# list of documents where each document is a list of words.
docTitle={}			# docTitle is dictionay where index is key and title is value.
docTitle_inv={}		# docTitle_inv is a dictionay where key is title and value is index.

for x in xrange(1,numberOfDocuments+1):
	document=open(directory+"/"+str(x)+".txt")
	docLines=document.readlines()		# break the document in a list of lines.
	document.close()
	docTerms=[]
	for i in xrange(0,len(docLines)):		# from first line to last line.
		if i==0:		# first line is title of the document
			docTitle[x-1]=docLines[i].strip()
			docTitle_inv.setdefault(docLines[i].strip(), len(docTitle_inv))
			pass
		pattern = re.compile(r'\W+')
		lineWords = pattern.split(docLines[i])		# break the line in a list of words, split by space.
		for j in xrange(0,len(lineWords)):
			if len(lineWords)>2:
				docTerms.append(lineWords[j].lower())		# store all the words of document in the single list of document words.
				pass
			pass
		pass
	documents.append(docTerms)		# store a document as set of words in the list documents.
	pass
print '1'
col_ptr=[0]
row_ind=[]
data=[]
uniqueTerm={}
term_inv={}

for d in documents:
    for term in d:
        index = uniqueTerm.setdefault(term, len(uniqueTerm))
        term_inv[index]=term
        row_ind.append(index)
        data.append(1)
        pass
    col_ptr.append(len(row_ind))
    pass
print '2'
sparseMatrix = csc_matrix((data, row_ind, col_ptr), dtype=float).toarray()
print '3'
u, s, vt = svds(numpy.asarray(sparseMatrix), z, which = 'LM')
print '4'
s=scipy.linalg.diagsvd(s,z,z)
print '5'
us=numpy.dot(u,s)
print '6'
v = numpy.transpose(vt)
print '7'
vt_s=numpy.dot(v,s)
print '8'
inverseOf_s=numpy.linalg.inv(s)
print '9'
tranposeOf_u=numpy.transpose(u)
print '10'

# part-1: return similar term ######################################

term_in=open(term_in,'r+')		# read terms from term_in file
inTerms=term_in.readlines()
term_in.close()
outSimilarTerms=[]		# list of similart terms to terms in input file of terms.
for l in xrange(0,len(inTerms)):
	inpTerm=inTerms[l].strip().lower()
	if(len(inpTerm)>2):
		stringIndex=uniqueTerm[inpTerm]
		similarityList={}
		for x in xrange(0,len(us)):
			similarity = spatial.distance.cosine(us[x],us[stringIndex])			# 1 - cosine function gives similarity between two terms.
			similarityList[x]=similarity
			pass
		print '11'
		sorted_x = sorted(similarityList.items(), key=operator.itemgetter(1))		# sort the list in increasing order of similarity between two terms.
		print '12'
		n=0
		similarTerms=[]
		while n<k:		# return k similar terms
			similarIndex=sorted_x[n][0]
			similarTerms.append(term_inv[similarIndex]+';\t')
			n=n+1
			pass
		outSimilarTerms.append(similarTerms)
		pass
	pass
print '13'
result=""
for x in outSimilarTerms:
	for terms in x:
		result += terms
	result+='\n'

g = open(term_out,"w")
g.write(result)
g.close()

print '14'
# part-2: return similar documents ########################################

doc_in=open(doc_in,'r+')

inDocs=doc_in.readlines()
doc_in.close()
outDocs=[]		# list of documents which is similar to given documents
for l in xrange(0,len(inDocs)):
	inpDoc=(inDocs[l]).strip()
	titleIndex=docTitle_inv[inpDoc]
	similarityList2={}
	for x in xrange(0,len(vt_s)):
		similarity2 = spatial.distance.cosine(vt_s[x],vt_s[titleIndex])
		similarityList2[x]=similarity2
		pass

	sortedSimilarityList = sorted(similarityList2.items(), key=operator.itemgetter(1))

	m=0
	similarDocuments=[]
	while m<k:		# return k similar documents
		tIndex=sortedSimilarityList[m][0]
		similarDocuments.append(docTitle[tIndex]+';\t')
		m=m+1
		pass
	outDocs.append(similarDocuments)
	pass

result=""
for x in outDocs:
	for terms in x:
		result += terms
	result+='\n'


g = open(doc_out,"w")
g.write(result)
g.close()
# part-3: return documents according to given query ######################
# m=(inv(s)*(tranpose(t))*q)

query_in=open(query_in,'r+')
inQuerys=query_in.readlines()
query_in.close()
outResult=[]
for l in xrange(0,len(inQuerys)):
	query=(inQuerys[l].lower()).strip()
	pattern = re.compile(r'\W+')
	setOfWords = pattern.split(query)
	q = [0]*len(uniqueTerm)

	for x in xrange(0,len(setOfWords)):
		if len(setOfWords[x])>2:
			i=uniqueTerm[setOfWords[x]]
			q[i]=1
			pass
		pass

	rowMatrix_M=numpy.dot(inverseOf_s,numpy.dot(tranposeOf_u,q))

	similarityList3={}
	for x in xrange(0,len(vt_s)):
		similarity3 = spatial.distance.cosine(vt_s[x],rowMatrix_M)
		similarityList3[x]=similarity3
		pass

	sortedSimilarityList3 = sorted(similarityList3.items(), key=operator.itemgetter(1))

	m=0
	similarDocuments3=[]
	while m<k:		# return top k results
		rIndex=sortedSimilarityList3[m][0]
		similarDocuments3.append(docTitle[rIndex]+";\t")
		m=m+1
		pass
	outResult.append(similarDocuments3)
	pass

result=""
for x in outResult:
	for terms in x:
		result += terms
	result+='\n'

g = open(query_out,"w")
g.write(result)
g.close()

