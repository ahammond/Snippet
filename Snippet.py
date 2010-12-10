#!/usr/bin/env python
# encoding: ascii

'''
Snippet.py

Created by Andrew Hammond on 2010-11-08.
Copyright (c) 2010 Andrew Hammond. All rights reserved.

Find the most relevant snippet for a document and highlight all query terms that appear in the snippet.

Since I do not know what modules are standard at Yelp, I'm sticking to python 2.6 standard stuff,
and libraries that happen to be included on my Mac (running osx 10.6.4).
I would generally use a mocking library such as Mock for UTs,
but I figured this had better run on anyone's machine.

Algorithm: 
I tried my hand at a custom algorithm and wasn't happy with the results,
so I've impelmented Smith-Waterman.
http://en.wikipedia.org/wiki/Smith-Waterman_algorithm
It varies from the standard approach since instead of matching amino acids, I'm matching words.
Naturally this means that I'm not using a BLOSUM matrix for weighted partial matches 
but have instead gone with a simple match / don't match approach.

Definitions:
Document - The input to be searched.

Query - The input to search for.

Snippet - A section of a document. A snippet is a string that looks like the following.
    __snippetMaxHead__ + local match terms + __snippetMaxTail__
'''

import re
import unittest
from collections import deque

# Initially, we won't worry much about normalizing the input. 
# TODO: Normalize tokens more intelligently
# like for example mapping "greasewheel" to pizza, or brewskie to beer.
__tokenNormalizer__ = lambda x : x.lower()

def highlight_doc(doc, query):
    '''
    Args:
    doc - String that is a document to be highlighted
    query - String that contains the search query
    Returns:
    The the most relevant snippet with the query terms highlighted.
    '''
    return str(Snippet(doc, query))

class Highlight_DocTests(unittest.TestCase):
    def setUp(self):
        self.d = 'An example document with some fun and interesting words in it.'
        self.q = 'document fun and words'

    def testBasic(self):
        self.assertEqual(
                highlight_doc(self.d, self.q),
                '... document with some [[HIGHLIGHT]]fun and[[ENDHIGHLIGHT]] interesting [[HIGHLIGHT]]words[[ENDHIGHLIGHT]] in it.')


class Token:
    '''A class to represent tokens and their offset into the string from which they were generated.'''
    def __init__(self, string, location, endLocation):
        self.__string__ = string
        self.__location__ = location
        self.__endLocation__ = endLocation

    def __repr__(self):
        return 'Token(\'%s\', %d, %d)' % (self.__string__, self.__location__, self.__endLocation__)

    def __cmp__(self, other):
        return cmp((self.__string__, self.__location__, self.__endLocation__),
                   (other.__string__, other.__location__, other.__endLocation__))

    def string(self):
        return self.__string__

    def location(self):
        return self.__location__

    def endLocation(self):
        return self.__endLocation__

class TokenTests(unittest.TestCase):
    def setUp(self):
        pass

    def testBasic(self):
        t = Token('a string', 24, 45)
        self.assertEqual(t.string(), 'a string')
        self.assertEqual(t.location(), 24)
        self.assertEqual(t.endLocation(), 45)


class Tokenized:
    '''A class to represent parsed, normalized tokens within a document or query.
    '''
    def __init__(self, document):
        '''Given a string, create a TokenMap object.
        '''
        self.document = document
        # a token is a series of alphanumeric characters or apostrophes not separated by whitespace or other punctuation
        self.__tokenizer__ = re.compile(r'([\w\']+)', re.UNICODE)

    def __str__(self):
        return self.list.__str__()

    def wordCount(self):
        if '__wordCount__' not in dir(self):
            self.list()
        return self.__wordCount__

    def map(self):
        '''Accessor function for a token map.
        The token map is a dictionary that maps each normalized token to a list of it's locations in the document.
        For example the map for 'A basic document'
        '''
        if '__map__' not in dir(self):
            self.__map__ = {}
            i = 0
            for token in self.__tokenizer__.finditer(self.document):
                name = __tokenNormalizer__(token.expand('\\1'))
                if self.__map__.has_key(name):
                    self.__map__[name].append(i)
                else:
                    self.__map__[name] = [i]
                i += 1
            self.__wordCount__ = i
        return self.__map__

    def list(self):
        '''Accessor function for a token list.
        The token list is a simple list of Tokens.
        '''
        if '__list__' not in dir(self):
            self.__list__ = []
            for tokenMatch in self.__tokenizer__.finditer(self.document):
                self.__list__.append(Token(__tokenNormalizer__(tokenMatch.expand('\\1')), tokenMatch.start(1), tokenMatch.end(1)))
            self.__wordCount__ = len(self.__list__)
        return self.__list__

class TokenizedTests(unittest.TestCase):
    def setUp(self):
        pass

    def testMapBasic(self):
        self.assertEqual(Tokenized('A basic document.').map(), {'a': [0], 'document': [2], 'basic': [1]})

    def testMapNormalization(self):
        self.assertEqual(Tokenized('A').map(), {'a': [0]})

    def testMapApostrophe(self):
        self.assertEqual(Tokenized("The apostrophe's usage isn't among the class' many issues.").map(),
                {'among': [4], "apostrophe's": [1], "class'": [6], 'issues': [8], 'many': [7], 'usage': [2], 'the': [0, 5], "isn't": [3]}
            )

    def testMapPunctuation(self):    # Note: no apostrophe or underscore
        self.assertEqual(Tokenized('!@#$%^&*()-=+\\|[]{};:"?/.>,<`~ ').map(), {})

    def testMapMultipleInstance(self):
        self.assertEqual(Tokenized('A, a and another a!').map(), {'a': [0, 1, 4], 'and': [2], 'another': [3]})

    def testListBasic(self):
        self.assertEqual(Tokenized('A basic query').list(),
            [Token('a', 0, 1), Token('basic', 2, 7), Token('query', 8, 13)])

    def testListMultipleInstance(self):
        self.assertEqual(Tokenized('A, a and another a!').list(),
            [Token('a', 0, 1), Token('a', 3, 4), Token('and', 5, 8), Token('another', 9, 16), Token('a', 17, 18)])

    def testWordCount(self):
        self.assertEqual(Tokenized('This is a six word sentence.').wordCount(), 6)


class Greatest:
    '''Given a set of inputs, retain and return the greatest.'''
    def __init__(self, initialValue):
        self.__currentValue__ = initialValue

    def append(self, nextValue):
        if nextValue > self.__currentValue__:
            self.__currentValue__ = nextValue
        return self.__currentValue__

    def value(self):
        return self.__currentValue__

class GreatestTests(unittest.TestCase):
    def setUp(self):
        self.g = Greatest(0)

    def testInit(self):
        self.assertEqual(self.g.value(), 0)

    def testAppendLarger(self):
        self.g.append(5)
        self.assertEqual(self.g.value(), 5)

    def testAppendSmaller(self):
        self.g.append(-5)
        self.assertEqual(self.g.value(), 0)


class SmithWatermanPoint:
    def __init__(self, weight, source):
        self.__weight__ = weight
        self.__source__ = source

    def __cmp__(self, other):
        return cmp(self.__weight__, other.__weight__)

    def __repr__(self):
        return '(%d, \'%s\')' % (self.__weight__, self.__source__)

    def weight(self):
        return self.__weight__

    def source(self):
        return self.__source__

class SmithWatermanPointTest(unittest.TestCase):
    def setUp(self):
        self.p = SmithWatermanPoint(8, 'i')

    def testWeight(self):
        self.assertEqual(self.p.weight(), 8)

    def testSource(self):
        self.assertEqual(self.p.source(), 'i')

    def testCmp(self):
        smallerPoint = SmithWatermanPoint(4, 'd')
        self.assertTrue(smallerPoint < self.p)


class SmithWatermanMatrix:
    weightMatch  =  2        # a match is worth 2 points
    weightMiss   = -1        # a mismatch costs 1
    dereferenceNode = {
            'm' : lambda x: (x[0]-1, x[1]-1),           # match or miss, go diagonally left and up
            'i' : lambda x: (x[0]  , x[1]-1),           # inserted a token, go left
            'd' : lambda x: (x[0]-1,   x[1])            # deleted a token, go up
        }

    def __init__(self, document, query):
        '''Given a document and a query as Tokenized objects, initialize a matrix of tupels to zero.'''
        self.document = document.list()
        self.query = query.list()
        self.documentLen = document.wordCount() + 1
        self.queryLen = query.wordCount() + 1
        self.__matrix__ = [0] * self.documentLen
        for d in xrange(0, self.documentLen):
            self.__matrix__[d] = [0] * self.queryLen
            for q in xrange(0, self.queryLen):
                self.__matrix__[d][q] = SmithWatermanPoint(0,' ')
        self.__highestWeight__ = SmithWatermanPoint(0, ' ')
        self.__highestLocation__ = (0,0)

    def __str__(self):
        '''A textual representation of the SW matrix.
        Across the top we have the query, and down the side, we have the document.
        A more traditional representation have the query across the top.
        '''
        documentOffset = max(map(lambda a: len(a.string()), self.document))    # find the longest term in the document
        queryOffset = max(map(lambda a: len(a.string()), self.query))    + 1    # find longest term in the search
        # If the largest query term is smaller than the len(r", (1,'m')"), default to that length.
        # Obviously this won't work very well for weights beyond 9 because it assumes a single digit.
        # TODO: support arbitrary weights
        if queryOffset < 10:
            queryOffset = 10
        s = ' ' * documentOffset                                         # header string
        # first query term is the "-" term (no match)
        s += ' ' * (queryOffset - 2) + '-'
        for q in self.query:
            s += ' ' * (queryOffset - len(q.string())) + q.string()
        s += "\n"
        for i in xrange(len(self.__matrix__)):
            d = self.document[i-1].string() if i > 0 else '-'
            s += ' ' * (documentOffset - len(d)) + d + str(self.__matrix__[i]) + " %d\n" % i
        return s

    #TODO: render to html for better readability

    def dimensions(self):
        return (self.documentLen, self.queryLen)

    def highestWeight(self):
        self.heat()
        return self.__highestWeight__

    def highestLocation(self):
        self.heat()
        return self.__highestLocation__

    def __matchWeight__(self, d, q):
        '''Boolean text match.
        '''
        # TODO: A more elegant match weighting algorithm here might be appropriate.
        # Maybe a BLOSUM type approach based on the kind of language involved?
        # For example, you could give a partial match value from pizza to pie.
        # This doesn't of course address the issue of mapping from one term to phrasal terms.
        return self.weightMatch if self.document[d-1].string() == self.query[q-1].string() else self.weightMiss

    def heat(self):
        '''An accessor function that generates the SW heat matrix.
        The matrix is initialized to zero elsewhere.
        Then, going left to right, from top to bottome and starting at the 1th index (not the 0th)
        We take the greatest of 0, the diagonally
        '''
        if '__hasBeenHeated__' not in dir(self):
            m = self.__matrix__
            for d in xrange(1, self.documentLen):                   # leave the 0th index elements at 0: guarantee
                for q in xrange(1, self.queryLen):                  # optimalPath search below finds an end point
                    match = self.__matchWeight__(d,q)
                    greatest = Greatest(SmithWatermanPoint(0, ' ')) # Omit this for a global match rather than local
                    greatest.append(SmithWatermanPoint(m[d - 1][q-1].weight() + match, 'm'))
                    greatest.append(SmithWatermanPoint(m[d - 1][q].weight() + match, 'd'))
                    greatest.append(SmithWatermanPoint(m[d][q - 1].weight() + match, 'i'))
                    self.__matrix__[d][q] = greatest.value()
                    if greatest.value() > self.__highestWeight__:
                        self.__highestWeight__ = greatest.value()
                        self.__highestLocation__ = (d,q)
            self.__hasBeenHeated__ = 1

    def optimalPath(self):
        if '__path__' not in dir(self):
            self.heat()            # can't path an un-heated matrix.
            # starting at the highest location, path the matrix first up and left, then down and right.
            self.__path__ = list()
            currentAddress = self.__highestLocation__
            currentPoint = self.__matrix__[currentAddress[0]][currentAddress[1]]
            while currentPoint.weight() > 0:
                if currentPoint.source() == 'm':
                    self.__path__.insert(0, currentAddress[0])
                currentAddress = self.dereferenceNode[currentPoint.source()](currentAddress)
                currentPoint = self.__matrix__[currentAddress[0]][currentAddress[1]]
        return self.__path__

class SmithWatermanMatrixTests(unittest.TestCase):
    def setUp(self):
        self.m = SmithWatermanMatrix(
            Tokenized('I like my deep dish pizza. Oh yes I do. Pizza!'),    # 11 words
            Tokenized('Deep dish pizza')                                    # 3 words
        )

    def testDimensions(self):
        self.assertEqual(self.m.dimensions(), (12, 4))

    def testStr(self):
        self.assertEqual(str(self.m), "             -      deep      dish     pizza\n" +
                                      "    -[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 0\n" +
                                      "    i[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 1\n" +
                                      " like[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 2\n" +
                                      "   my[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 3\n" +
                                      " deep[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 4\n" +
                                      " dish[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 5\n" +
                                      "pizza[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 6\n" +
                                      "   oh[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 7\n" +
                                      "  yes[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 8\n" +
                                      "    i[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 9\n" +
                                      "   do[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 10\n" +
                                      "pizza[(0, ' '), (0, ' '), (0, ' '), (0, ' ')] 11\n"
        )

    def testHighestWeightAndLocation(self):
        self.assertEqual(self.m.highestWeight().weight(), 6)
        self.assertEqual(self.m.highestLocation(), (6,3))

    def testOptimalPath(self):
        self.assertEqual(self.m.optimalPath(), [4, 5, 6])

    def testOptimalPathComplex(self):
        m=SmithWatermanMatrix(
            Tokenized('I like pizza and beer together. I especially like the deep dish pepperoni pizza with a side salad and a deep dish of peppers.'),
            Tokenized('deep dish pizza')
        )
        self.assertEqual(m.dimensions(), (25,4))
        self.assertEqual(m.optimalPath(), [11, 12, 14])

        
class Snippet:
    '''A class to support the management and generation of snippets.'''

    __snippetMaxHead__ = 20
    __snippetMaxTail__ = 20
    __highlightStart__ = '[[HIGHLIGHT]]'
    __highlightEnd__   = '[[ENDHIGHLIGHT]]'

    def __init__(self, document, query):
        self.document = Tokenized(document)
        self.query = Tokenized(query)

    def matrix(self):
        '''Accessor function that constructs the SmithWaterman alignment matrix.
        For matrix[a][b], the document location is the a'th index and the query location is the b'th.
        Note that the 0th index for the SW matrix is always set to zero,
        so the matrix is effictively indexed from 1 to len() + 1.
        This leads to some unfortunate uglieness around indexing into the Tokenized arrays vs the SW matrix.
        '''
        if '__matrix__' not in dir(self):
            self.__matrix__ = SmithWatermanMatrix(self.document, self.query)
        return self.__matrix__

    def highlightSpans(self):
        '''Accessor function that constructs a list of tuples that represent the beginning and end of highlight sections.
        '''
        if '__highlights__' not in dir(self):
            self.__highlights__ = []
            terms = deque(self.matrix().optimalPath())
            lastTerm = None
            try:
                # remember, matrix indexing is +1 from Tokenized list indexing
                t = terms.popleft() - 1
                while 1:
                    self.__highlights__.append([t, t])          # start of a series
                    t = terms.popleft() - 1
                    while self.__highlights__[-1][1] + 1 == t:  # advance through the list of terms which are sequential
                        self.__highlights__[-1][1] = t
                        t = terms.popleft() - 1
            except IndexError:
                pass
        return self.__highlights__

    def startIndexFromTerm(self, termNumber):
        '''Given a term in the document.list(), return it's start index
        '''
        return self.document.list()[termNumber].location()

    def endIndexFromTerm(self, termNumber):
        return self.document.list()[termNumber].endLocation()

    def __str__(self):
        '''Return a highlited snippet.
        '''
        spans = deque(self.highlightSpans())
        d = self.document.document
        r = ''
        cursor = 0
        # TODO: should this instead be breaking on work boundaries?
        if len(spans) > 0:
            peekStart, peekEnd = spans.popleft()
            spans.appendleft((peekStart, peekEnd))
            cursor = self.startIndexFromTerm(peekStart) - self.__snippetMaxHead__
            if cursor > 0:
                r = '...'
            else:
                cursor = 0
        while len(spans):
            startTerm, endTerm = spans.popleft()
            start = self.startIndexFromTerm(startTerm)
            end = self.endIndexFromTerm(endTerm)
            r += d[cursor:start] + self.__highlightStart__ + d[start:end] + self.__highlightEnd__
            cursor = end
        # TODO: like the head, should the tail break on word boundaries instead
        tailLength = cursor+self.__snippetMaxTail__
        if tailLength > len(d):
            r+=d[cursor:]
        else:
            r += d[cursor:tailLength] + '...'
        return r

class SnippetTests(unittest.TestCase):
    def setUp(self):    #      0       1    2    3     4    5  6         7
        self.s = Snippet('Little star\'s deep dish pizza sure is fantastic.', 'deep dish pizza')

    def testHighlightSpans(self):
        self.assertEqual(self.s.highlightSpans(), [[2,4]])

    def testHighlightSpansComplex(self):
        s = Snippet('A big complicated document with some complicated stuff in it.', 'with some stuff')
        self.assertEqual(s.highlightSpans(), [[4,5],[7,7]])

    def testIndexFromTerm(self):
        self.assertEqual(self.s.startIndexFromTerm(3), 19)

    def testStr(self):
        self.assertEqual(str(self.s), 'Little star\'s [[HIGHLIGHT]]deep dish pizza[[ENDHIGHLIGHT]] sure is fantastic.')

    def testStrComplex(self):
        s = Snippet('A big complicated document with some complicated stuff in it.', 'with some stuff')
        self.assertEqual(str(s), '...omplicated document [[HIGHLIGHT]]with some[[ENDHIGHLIGHT]] complicated [[HIGHLIGHT]]stuff[[ENDHIGHLIGHT]] in it.')

    def testStrLongTail(self):
        s = Snippet('A big complicated document with some complicated stuff in it.', 'big')
        self.assertEqual(str(s), 'A [[HIGHLIGHT]]big[[ENDHIGHLIGHT]] complicated documen...')

if __name__ == '__main__':
    unittest.main()
