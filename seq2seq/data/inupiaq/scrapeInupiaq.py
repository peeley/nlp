from  bs4 import BeautifulSoup
import pandas as pd
import requests, re, random
from unidecode import unidecode

def scrapePage(url, book, chapter, langCode, exclusions):
    try:
        request = requests.get(url)
    except:
        raise ConnectionError('Unable to connect to', url)
    bibleText = request.text
    bibleSoup = BeautifulSoup(bibleText, 'html5lib')
    # inupiaq langCode = esi, english langCode = eng
    column = bibleSoup.findAll('div', {'data-iso6393': langCode})
    paragraphs = column[0].findAll('span', {'data-usfm': ''})

    skipFlag = True
    lines = {}

    for paragraph in paragraphs:
        if paragraph['class'] == ['label'] and '#' not in paragraph.text:
            verseNumber = paragraph.text
            if '-' in verseNumber :
                numbers = verseNumber.split('-')
                exclusionRange = range(int(numbers[0]), int(numbers[1])+1)
                for i in exclusionRange:
                    exclusion = book + '.' + str(chapter) + '.' + str(i)
                    exclusions.append(exclusion)
                skipFlag = True
            elif ',' in verseNumber:
                numbers = verseNumber.split(',')
                exclusionRange = range(int(numbers[0]), int(numbers[1])+1)
                for i in exclusionRange:
                    exclusion = book + '.' + str(chapter) + '.' + str(i)
                    exclusions.append(exclusion)
                skipFlag = True
            else:
                verseName = book + '.' + str(chapter) + '.' + verseNumber
                skipFlag = False
        if paragraph['class'] == ['content'] and paragraph.text.strip() != '':
            if not skipFlag:
                try:
                    lines[verseName] = lines[verseName] + ' ' + paragraph.text
                except KeyError:
                    lines[verseName] = paragraph.text
            else:
                continue
    return lines, exclusions


def stripLine(line):
    line  = re.sub(u'[\u201c\u201d]','', line)
    line  = line.replace('"', '')
    line  = re.sub('([.,!?()])', r' \1 ', line)
    line  = re.sub('\s{2,}', ' ', line)
    return line

def transcribeLines(lines, exclusions, file, lang):
    for key in lines:
        if key in exclusions:
            print('\tSkipping verse {} in {}'.format(key, lang))
        else:
            print('\tWriting verse {} in {}'.format(key, lang))
            string = key + ': ' + lines[key] + '\n'
            file.write(stripLine(lines[key]) + '\n')


def scrapeBible():
    books = {'rut':4,'est':10,'oba':1,'jon':4, 'mat':28, 'mrk':16, 'luk':24, 'jhn':21, 'act':28, 'rom':16, '1co':16, '2co':13, 'gal':6, 'eph':6, 'php':4, 'col':4, '1th':5, '2th':3, '1ti':6, '2ti':4, 'tit':3, 'phm':1, 'heb':13, 'jas':5, '1pe':5, '2pe':3, '1jn':5, '2jn':1, '3jn':1, 'jud':1, 'rev': 22}

    engLines = {}
    ipqLines = {}
    engFileName = 'bible_eng'
    ipqFileName = 'bible_ipq'
    engFile     = open(engFileName, 'w+')
    ipqFile     = open(ipqFileName, 'w+')
    engValFile  = open(engFileName + '_val', 'w+')
    ipqValFile  = open(ipqFileName + '_val', 'w+')
    engTestFile = open(engFileName + '_test', 'w+')
    ipqTestFile = open(ipqFileName + '_test', 'w+')
    validate = False
    test = False
    for book in books:
        for chapter in range(1, books[book]+1):
            try:
                ipqBibleURL = 'https://www.bible.com/bible/1415/' + book + '.' + str(chapter) 
                engBibleURL = 'https://www.bible.com/bible/392/'  + book + '.' + str(chapter)
                print('\nWriting {} chapter {}:'.format(book, chapter))

                exclusions = []
                ipqLines, exclusions = scrapePage(ipqBibleURL, book, chapter, 'esi', exclusions)
                engLines, exclusions = scrapePage(engBibleURL, book, chapter, 'eng', exclusions)
                prob = random.random()
                if prob < .03:
                    transcribeLines(engLines, exclusions, engTestFile, 'eng')
                    transcribeLines(ipqLines, exclusions, ipqTestFile, 'ipq')
                elif prob > .03 and prob < .06:
                    transcribeLines(engLines, exclusions, engValFile, 'eng')
                    transcribeLines(ipqLines, exclusions, ipqValFile, 'ipq')
                else:
                    transcribeLines(engLines, exclusions, engFile, 'eng')
                    transcribeLines(ipqLines, exclusions, ipqFile, 'ipq')
            except ValueError:
                print('\nSkipping {} chapter {}:'.format(book, chapter))
                continue
    engFile.close()
    ipqFile.close()
    engTestFile.close()
    ipqTestFile.close()
    print('\nData written to file.\n')

if __name__ == '__main__':
    scrapeBible()




