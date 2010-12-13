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

class HighlightDocTests(unittest.TestCase):
    def setUp(self):
        self.d = 'An example document with some fun and interesting words in it.'
        self.q = 'document fun and words'

    def test_go_right(self):
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

    def start_index(self):
        return self.__location__

    def end_index(self):
        return self.__endLocation__

class TokenTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_go_right(self):
        t = Token('a string', 24, 45)
        self.assertEqual(t.string(), 'a string')
        self.assertEqual(t.start_index(), 24)
        self.assertEqual(t.end_index(), 45)


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

    def word_count(self):
        if '__wordCount__' not in dir(self):
            self.list()
        return self.__wordCount__

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

    def test_list_go_right(self):
        self.assertEqual(Tokenized('A basic query').list(),
            [Token('a', 0, 1), Token('basic', 2, 7), Token('query', 8, 13)])

    def test_list_multiple_instance(self):
        self.assertEqual(Tokenized('A, a and another a!').list(),
            [Token('a', 0, 1), Token('a', 3, 4), Token('and', 5, 8), Token('another', 9, 16), Token('a', 17, 18)])

    def testWordCount(self):
        self.assertEqual(Tokenized('This is a six word sentence.').word_count(), 6)


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

    def test_init_go_right(self):
        self.assertEqual(self.g.value(), 0)

    def test_append_larger(self):
        self.g.append(5)
        self.assertEqual(self.g.value(), 5)

    def test_append_smaller(self):
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

    def test_weight(self):
        self.assertEqual(self.p.weight(), 8)

    def test_source(self):
        self.assertEqual(self.p.source(), 'i')

    def test_cmp(self):
        smallerPoint = SmithWatermanPoint(4, 'd')
        self.assertTrue(smallerPoint < self.p)


class SmithWatermanMatrix:
    weight_match  =  2        # a match is worth 2 points
    weight_miss   = -1        # a mismatch costs 1
    dereferenceNode = {
            'm' : lambda x: (x[0]-1, x[1]-1),           # match or miss, go diagonally left and up
            'i' : lambda x: (x[0]  , x[1]-1),           # inserted a token, go left
            'd' : lambda x: (x[0]-1,   x[1])            # deleted a token, go up
        }

    def __init__(self, document, query):
        '''Given a document and a query as Tokenized objects, initialize a matrix of tupels to zero.'''
        self.document = document.list()
        self.query = query.list()
        self.document_length = document.word_count() + 1
        self.query_length = query.word_count() + 1
        self.__matrix__ = [0] * self.document_length
        for d in xrange(0, self.document_length):
            self.__matrix__[d] = [0] * self.query_length
            for q in xrange(0, self.query_length):
                self.__matrix__[d][q] = SmithWatermanPoint(0,' ')
        self.__highest_weight__ = SmithWatermanPoint(0, ' ')
        self.__highest_location__ = (0,0)

    def __str__(self):
        '''A textual representation of the SW matrix.
        Across the top we have the query, and down the side, we have the document.
        A more traditional representation have the query across the top.
        '''
        document_offset = max(map(lambda a: len(a.string()), self.document))    # find the longest term in the document
        query_offset = max(map(lambda a: len(a.string()), self.query))    + 1    # find longest term in the search
        # If the largest query term is smaller than the len(r", (1,'m')"), default to that length.
        # Obviously this won't work very well for weights beyond 9 because it assumes a single digit.
        # TODO: support arbitrary weights
        if query_offset < 10:
            query_offset = 10
        s = ' ' * document_offset                                         # header string
        # first query term is the "-" term (no match)
        s += ' ' * (query_offset - 2) + '-'
        for q in self.query:
            s += ' ' * (query_offset - len(q.string())) + q.string()
        s += "\n"
        for i in xrange(len(self.__matrix__)):
            d = self.document[i-1].string() if i > 0 else '-'
            s += ' ' * (document_offset - len(d)) + d + str(self.__matrix__[i]) + " %d\n" % i
        return s

    #TODO: render to html for better readability

    def dimensions(self):
        return self.document_length, self.query_length

    def highestWeight(self):
        self.heat()
        return self.__highest_weight__

    def highestLocation(self):
        self.heat()
        return self.__highest_location__

    def __matchWeight__(self, d, q):
        '''Boolean text match.
        '''
        # TODO: A more elegant match weighting algorithm here might be appropriate.
        # Maybe a BLOSUM type approach based on the kind of language involved?
        # For example, you could give a partial match value from pizza to pie.
        # This doesn't of course address the issue of mapping from one term to phrasal terms.
        return self.weight_match if self.document[d-1].string() == self.query[q-1].string() else self.weight_miss

    def heat(self):
        '''An accessor function that generates the SW heat matrix.
        The matrix is initialized to zero elsewhere.
        Then, going left to right, from top to bottome and starting at the 1th index (not the 0th)
        We take the greatest of 0, the diagonally
        '''
        if '__heated__' not in dir(self):
            m = self.__matrix__
            for d in xrange(1, self.document_length):                   # leave the 0th index elements at 0: guarantee
                for q in xrange(1, self.query_length):                  # optimalPath search below finds an end point
                    match = self.__matchWeight__(d,q)
                    greatest = Greatest(SmithWatermanPoint(0, ' ')) # Omit this for a global match rather than local
                    greatest.append(SmithWatermanPoint(m[d - 1][q-1].weight() + match, 'm'))
                    greatest.append(SmithWatermanPoint(m[d - 1][q].weight() + match, 'd'))
                    greatest.append(SmithWatermanPoint(m[d][q - 1].weight() + match, 'i'))
                    self.__matrix__[d][q] = greatest.value()
                    if greatest.value() > self.__highest_weight__:
                        self.__highest_weight__ = greatest.value()
                        self.__highest_location__ = (d,q)
            self.__heated__ = 1

    def optimalPath(self):
        if '__path__' not in dir(self):
            self.heat()            # can't path an un-heated matrix.
            # starting at the highest location, path the matrix first up and left, then down and right.
            self.__path__ = list()
            current_address = self.__highest_location__
            current_point = self.__matrix__[current_address[0]][current_address[1]]
            while current_point.weight() > 0:
                if current_point.source() == 'm':
                    self.__path__.insert(0, current_address[0])
                current_address = self.dereferenceNode[current_point.source()](current_address)
                current_point = self.__matrix__[current_address[0]][current_address[1]]
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

    __snippet_max_head__ = 20
    __snippet_max_tail__ = 20
    __highlight_start__ = '[[HIGHLIGHT]]'
    __highlight_end__   = '[[ENDHIGHLIGHT]]'

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

    def highlight_spans(self):
        '''Accessor function that constructs a list of tuples that represent the beginning and end of highlight sections.
        '''
        if '__highlights__' not in dir(self):
            self.__highlights__ = []
            terms = deque(self.matrix().optimalPath())
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

    def start_index(self, termNumber):
        '''Given a term in the document.list(), return it's start index
        '''
        return self.document.list()[termNumber].start_index()

    def end_index(self, termNumber):
        '''Geven a term in the document.list(), return it's end index
        '''
        return self.document.list()[termNumber].end_index()

    def has_match(self):
        return 0 < len(self.matrix().optimalPath())

    def __str__(self):
        '''Return a highlited snippet.
        '''
        spans = deque(self.highlight_spans())
        d = self.document.document
        r = ''
        cursor = 0
        # TODO: should this instead be breaking on work boundaries?
        if len(spans) > 0:
            peekStart, peekEnd = spans.popleft()
            spans.appendleft((peekStart, peekEnd))
            cursor = self.start_index(peekStart) - self.__snippet_max_head__
            if cursor > 0:
                r = '...'
            else:
                cursor = 0
        while len(spans):
            startTerm, endTerm = spans.popleft()
            start = self.start_index(startTerm)
            end = self.end_index(endTerm)
            r += d[cursor:start] + self.__highlight_start__ + d[start:end] + self.__highlight_end__
            cursor = end
        # TODO: like the head, should the tail break on word boundaries instead
        tailLength = cursor+self.__snippet_max_tail__
        if tailLength > len(d):
            r+=d[cursor:]
        else:
            r += d[cursor:tailLength] + '...'
        return r

class SnippetTests(unittest.TestCase):
    def setUp(self):    #      0       1    2    3     4    5  6         7
        self.s = Snippet('Little star\'s deep dish pizza sure is fantastic.', 'deep dish pizza')

    def test_has_match_True(self):
        self.assertTrue(self.s.has_match())

    def test_has_match_False(self):
        s = Snippet('No match', 'foo')
        self.assertFalse(s.has_match())

    def test_highlight_spans_go_right(self):
        self.assertEqual(self.s.highlight_spans(), [[2,4]])

    def test_highlight_spans_go_right_complex(self):
        s = Snippet('A big complicated document with some complicated stuff in it.', 'with some stuff')
        self.assertEqual(s.highlight_spans(), [[4,5],[7,7]])

    def test_start_index(self):
        self.assertEqual(self.s.start_index(3), 19)

    def test_end_index(self):
        self.assertEqual(self.s.end_index(3), 23)

    def test_str(self):
        self.assertEqual(str(self.s), 'Little star\'s [[HIGHLIGHT]]deep dish pizza[[ENDHIGHLIGHT]] sure is fantastic.')

    def test_str_complex(self):
        s = Snippet('A big complicated document with some complicated stuff in it.', 'with some stuff')
        self.assertEqual(str(s), '...omplicated document [[HIGHLIGHT]]with some[[ENDHIGHLIGHT]] complicated [[HIGHLIGHT]]stuff[[ENDHIGHLIGHT]] in it.')

    def test_str_long_tail(self):
        s = Snippet('A big complicated document with some complicated stuff in it.', 'big')
        self.assertEqual(str(s), 'A [[HIGHLIGHT]]big[[ENDHIGHLIGHT]] complicated documen...')

if __name__ == '__main__':
    unittest.main()
