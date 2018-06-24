from  bs4 import BeautifulSoup
import pandas as pd
import requests, re


books = {'RUT':4,'EST':10,'OBA':1,'JON':4, 'MAT':28, 'MRK':16, 'LUK':24, 'JHN':21, 'ACT':28, 'ROM':16, '1CO':16, '2CO':13, 'GAL':6, 'EPH':6, 'PHP':4, 'COL':4, '1TH':5, '2TH':3, '1TI':6, '2TI':4, 'TIT':3, 'PHM':1, 'HEB':13, 'JAS':5, '1PE':5, '2PE':3, '1JN':5, '2JN':1, '3JN':1, 'JUD':1, 'REV': 22}

chapterIndex = 0
index = chapterIndex
dataFrame = pd.DataFrame(columns = ['english', 'inupiaq'])
for book in books:
    for chapter in range(1, books[book]):
        bibleURL = 'https://www.bible.com/bible/1415/' + book + '.' + str(chapter) + '.net?parallel=107' 
        print(book,' chapter', chapter)
        bibleRequest = requests.get(bibleURL)
        bibleText = bibleRequest.text
        bibleSoup = BeautifulSoup(bibleText, 'lxml')

        inupiaqColumn = bibleSoup.findAll('div', {'data-iso6393':'esi'})
        englishColumn = bibleSoup.findAll('div', {'data-iso6393':'eng'})
        inupiaqParagraphs = inupiaqColumn[0].findAll('div', {'class':'p'})
        englishParagraphs = englishColumn[0].findAll('div', {'class':'p'})

        for paragraph in inupiaqParagraphs:
            inupiaqVerses = paragraph.find_all('span', {'class': re.compile(r'verse ')})
            for verse in inupiaqVerses:
                ipqLineElements = verse.find_all('span', {'class' : 'content'})
                ipqLine = ''
                for element in ipqLineElements:
                    ipqLine += element.text
                if len(ipqLine) > 1:
                    print('writing ipq at index, ', index)
                    dataFrame.loc[index, 'inupiaq'] = ipqLine
                    index += 1
        index = chapterIndex
        for paragraph in englishParagraphs:
            engVerses = paragraph.find_all('span', {'class': re.compile(r'verse ')})
            for verse in engVerses:
                engLineElements = verse.find_all('span', {'class' : 'content'})
                engLine = ''
                for element in engLineElements:
                    engLine += element.text
                if len(engLine) > 1:
                    dataFrame.loc[index, 'english'] = engLine
                    index += 1
        chapterIndex = index
print(dataFrame)
dataFrame.to_csv('bible.csv')

# TODO: Script can now download entire text, but line breaks cause duplicate verses in HTML 'p' tags.
