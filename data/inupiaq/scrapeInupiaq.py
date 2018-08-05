from  bs4 import BeautifulSoup
import pandas as pd
import requests, re


def scrapeParagraph(paragraphs, frame, column, index):
    for paragraph in paragraphs:
        verses = paragraph.find_all('span', {'class': re.compile(r'verse ')})
        for verse in verses:
            elements = verse.find_all('span', {'class' : 'content'})
            line = ''
            for element in elements:
                line += element.text
            '''
            sentences = line.split('.')
            for sentence in sentences:
                sentence = sentence.strip('"(“[”])’')
                print(sentence)
                if len(sentence.strip()) > 1:
                    frame.loc[index, column] = sentence
                    index += 1
            '''
            if len(line.strip()) > 1:
                frame.loc[index, column] = line 
                index += 1
    return index
def scrapeBible():
    books = {'RUT':4,'EST':10,'OBA':1,'JON':4, 'MAT':28, 'MRK':16, 'LUK':24, 'JHN':21, 'ACT':28, 'ROM':16, '1CO':16, '2CO':13, 'GAL':6, 'EPH':6, 'PHP':4, 'COL':4, '1TH':5, '2TH':3, '1TI':6, '2TI':4, 'TIT':3, 'PHM':1, 'HEB':13, 'JAS':5, '1PE':5, '2PE':3, '1JN':5, '2JN':1, '3JN':1, 'JUD':1, 'REV': 22}

    engIdx = 0
    ipqIdx = 0
    dataFrame = pd.DataFrame(columns = ['english', 'inupiaq'])
    #for book in books:
    book = 'RUT'
    for chapter in range(1, books[book]):
        bibleURL = 'https://www.bible.com/bible/1415/' + book + '.' + str(chapter) + '.net?parallel=392' 
        print(book, ' chapter', chapter)
        try:
            bibleRequest = requests.get(bibleURL)
        except:
            raise ConnectionError('Unable to connect to site ', bibleURL)
        bibleText = bibleRequest.text
        bibleSoup = BeautifulSoup(bibleText, 'html5lib')

        inupiaqColumn = bibleSoup.findAll('div', {'data-iso6393':'esi'})
        englishColumn = bibleSoup.findAll('div', {'data-iso6393':'eng'})
        inupiaqParagraphs = inupiaqColumn[0].findAll('div', {'class':'p'})
        englishParagraphs = englishColumn[0].findAll('div', {'class':'p'})
        ipqIdx = scrapeParagraph(inupiaqParagraphs, dataFrame, 'inupiaq', ipqIdx)
        engIdx = scrapeParagraph(englishParagraphs, dataFrame, 'english', engIdx)

    print(dataFrame)
    dataFrame.to_csv('bible.csv')

if __name__ == '__main__':
    scrapeBible()
# TODO: Script can now download entire text, but line breaks cause duplicate verses in HTML 'p' tags.
