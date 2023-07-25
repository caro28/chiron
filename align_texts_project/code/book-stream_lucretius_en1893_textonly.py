import re
import sys
from lxml import etree

## This code assumes that chunk boundaries tagged by <div> strictly
## contain blocks of text tagged by <p>. If you want to modify it for
## texts like Plato which have page-based chunking from a <milestone>
## tag, you'd need to figure out what to do with chunk boundaries
## occurring in the middle of a sentence. In that case, an acceptable
## naive solution might be to delay until the end of the paragraph and
## then print the chunk,paragraph pair at once.

'''
Code from David Smith
'''

class BookStream(object):
    def __init__(self):
        self.divs = []
        self.buf = ''
        self.inText = False
    def start(self, elem, attrib):
        match etree.QName(elem).localname:
            case 'text':
                self.inText = True
            case 'div':
                if attrib.get('type', '') == 'textpart':
                    level = attrib.get('subtype', '') + '=' + attrib.get('n', '')
                else:
                    level = ''
                self.divs.append(level)
                # cite = ','.join([x for x in self.divs if x != ''])
                # ##### comment out if statement to print without section markers #####
                # if cite != '':
                #     print('#@$%', cite, '#@$%\n')
            case 'note':
                # self.buf += '\n#@$%note#@$%'
                # self.buf += '\n'
                if attrib.get('type', '') == 'correspondsTo':
                    print("$$$$$$$$$$")
            # case 'title':
                # self.buf += '#@$%title#@$%'
                # self.buf += '\n'
                # self.inText = False
    def end(self, elem):
        match etree.QName(elem).localname:
            case 'div':
                self.divs.pop()
            # case 'head' | 'p':
                # print(self.buf.strip(), '\n#@$%paragraph end#@$%')
                ##### use line below instead to print without section markers #####
                # print(self.buf.strip(), '\n')
                # self.buf = ''
            ##### comment out case "note" section below to print without section markers #####
            # case 'note':
            #     # self.buf += '\n#@$%note_end#@$%'
            #     self.buf += '\n'
            # case 'title':
            #     self.buf += '\n'
            #### add page numbers? code below doesn't work #####
            # case 'head' | 'pb':
            #     print(self.buf.strip(), '\n')
                # self.buf = ''
    def data(self, data):
        if self.inText:
            self.buf += re.sub(r'\s+', ' ', re.sub(r'\-\n', '', data))
    def comment(self, text):
        pass
    def close(self):
        return "closed!"
    
if __name__ == '__main__':
    ns = {'': 'http://www.tei-c.org/ns/1.0'}
    file = "/home/craig.car/repos/chiron/align_texts_project/data/lucretius/lucretius_1893.xml"
    
    parser = etree.XMLParser(target = BookStream())
    
    result = etree.parse(file, parser)
