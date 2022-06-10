import textract
import docx2txt
import nltk
import re
from nltk.tokenize import sent_tokenize
text_doc = docx2txt.process('script.docx')
text_doc_replace = re.sub('\n', " ", text_doc)
print(text_doc_replace)
sent = sent_tokenize(text_doc_replace)
print(sent)
for i in sent:
    print(i)
