from  bs4 import BeautifulSoup
import pandas as pd
import requests, re


books = {'RUT':4,'EST':10,'OBA':1,'JON':4, 'MAT':28, 'MRK':16, 'LUK':24, 'JHN':21, 'ACT':28, 'ROM':16, '1CO':16, '2C0':13, 'GAL':6, 'EPH':6, 'PHP':4, 'COL':4, '1TH':5, '2TH':3, '1TI':6, '2TI':4, 'TIT':3, 'PHM':1, 'HEB':13, 'JAS':5, '1PE':5, '2PE':3, '1JN':5, '2JN':1, '3JN':1, 'JUD':1, 'REV': 22}
bibleURL = 'https://www.bible.com/bible/1415/JHN.1.INUPIAQ?parallel=1359'
bibleRequest = requests.get(bibleURL)
bibleText = bibleRequest.text
bibleSoup = BeautifulSoup(bibleText, 'lxml')

dataFrame = pd.DataFrame(columns = ['english', 'inupiaq'])
inupiaqColumn = bibleSoup.findAll('div', {'data-iso6393':'esi'})
englishColumn = bibleSoup.findAll('div', {'data-iso6393':'eng'})

inupiaqParagraphs = inupiaqColumn[0].findAll('div', {'class':'p'})
englishParagraphs = englishColumn[0].findAll('div', {'class':'p'})

index = 0
for paragraph in inupiaqParagraphs:
    inupiaqVerses = paragraph.find_all('span', {'class': re.compile(r'verse ')})
    for verse in inupiaqVerses:
        ipqLineElements = verse.find_all('span', {'class' : 'content'})
        ipqLine = ''
        for element in ipqLineElements:
            ipqLine += element.text
        if len(ipqLine) > 1:
            dataFrame['inupiaq'].loc[index] = ipqLine
        index += 1
print(dataFrame['inupiaq'])
index = 0
for paragraph in englishParagraphs:
    engVerses = paragraph.find_all('span', {'class': re.compile(r'verse ')})
    for verse in engVerses:
        engLineElements = verse.find_all('span', {'class' : 'content'})
        engLine = ''
        for element in engLineElements:
            engLine += element.text
        if len(engLine) > 1:
            dataFrame['english'].loc[index] = engLine
        index += 1
print(dataFrame['english'])
dataFrame.to_csv('inupiaq.csv')

# TODO: When printing columns of dataframe, all data is present but printing whole dataframe and saving to csv shows blank.
#       Rows of english/inupiaq columns are also inequal, perhaps there is something going on with verses and sentences not
#       being interchangable.
