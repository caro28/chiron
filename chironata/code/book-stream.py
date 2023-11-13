#!/usr/bin/env python

import re
import sys, os
from lxml import etree

## This code prints out one paragraph per line, with the associated
## citation and page number where available. It therefore doesn't
## handle the case of section breaks in the middle of paragraphs.

## div types that contain main text or not
goods = {'edition', 'translation'}
bads = {'commentary'}

class BookStream(object):
    def __init__(self):
        self.divs = [False]
        self.textparts = []
        self.print = [False]
        self.buf = ''
        self.cite = ''
        self.page = ''
    def start(self, elem, attrib):
        lname = etree.QName(elem).localname
        if lname == 'div':
            dtype = attrib.get('type', '')
            if dtype in goods:
                divOK = True
            elif dtype in bads:
                divOK = False
            else:
                divOK = self.divs[-1]
            self.divs.append(divOK)
            if attrib.get('type', '') == 'textpart':
                level = attrib.get('subtype', '') + '=' + attrib.get('n', '')
            else:
                level = ''
            self.textparts.append(level)
            self.cite = ','.join([x for x in self.textparts if x != ''])
        elif lname == 'p':
            self.print.append(self.divs[-1])
        elif lname == 'note':
            self.print.append(False)
        elif lname == 'pb':
            self.page = attrib.get('n', '')
    def end(self, elem):
        lname = etree.QName(elem).localname
        if lname == 'div':
            self.textparts.pop()
            self.divs.pop()
        elif lname == 'p':
            if self.print.pop():
                print('\t'.join([self.cite, self.page, re.sub(r'\s+', ' ', self.buf.strip())]))
                self.buf = ''
        elif lname == 'note':
            self.print.pop()
    def data(self, data):
        if self.print[-1]:
            self.buf += re.sub(r'\s+', ' ', re.sub(r'\-\n', '', data))
    def comment(self, text):
        pass
    def close(self):
        return "closed!"
    
if __name__ == '__main__':
    ns = {'': 'http://www.tei-c.org/ns/1.0'}
    
    parser = etree.XMLParser(target = BookStream())
    
    prefix = os.path.splitext(sys.argv[1])[0]
    with open(prefix+".par", 'w') as sys.stdout:
        result = etree.parse(sys.argv[1], parser)
