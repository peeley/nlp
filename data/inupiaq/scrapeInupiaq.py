from  bs4 import BeautifulSoup
import pandas as pd
import requests, re


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



def transcribeLines(lines, exclusions, frame, lang, idx):
    for key in lines:
        if key in exclusions:
            print('\tSkipping excluded verse {} in {}'.format(key, lang))
        else:
            print('\tWriting verse {} in {}'.format(key, lang))
            string = key + ': ' + lines[key] + '\n'
            frame.loc[idx, (lang+'Verse')] = lines[key]
            frame.loc[idx, (lang+'Num')]   = key
            idx += 1
    return frame


def scrapeBible():
    books = {'rut':4,'est':10,'oba':1,'jon':4, 'mat':28, 'mrk':16, 'luk':24, 'jhn':21, 'act':28, 'rom':16, '1co':16, '2co':13, 'gal':6, 'eph':6, 'php':4, 'col':4, '1th':5, '2th':3, '1ti':6, '2ti':4, 'tit':3, 'phm':1, 'heb':13, 'jas':5, '1pe':5, '2pe':3, '1jn':5, '2jn':1, '3jn':1, 'jud':1, 'rev': 22}

    engLines = {}
    ipqLines = {}
    dataFrame = pd.DataFrame(columns = ['engNum', 'engVerse', 'ipqNum', 'ipqVerse'])
    for book in books:
        for chapter in range(1, books[book]+1):
            try:
                ipqBibleURL = 'https://www.bible.com/bible/1415/' + book + '.' + str(chapter) 
                engBibleURL = 'https://www.bible.com/bible/392/'  + book + '.' + str(chapter)
                print('\nWriting {} chapter {}:'.format(book, chapter))

                idx = len(dataFrame)
                exclusions = []
                ipqLines, exclusions = scrapePage(ipqBibleURL, book, chapter, 'esi', exclusions)
                engLines, exclusions = scrapePage(engBibleURL, book, chapter, 'eng', exclusions)
                dataFrame = transcribeLines(engLines, exclusions, dataFrame, 'eng', idx)
                dataFrame = transcribeLines(ipqLines, exclusions, dataFrame, 'ipq', idx)
            except ValueError:
                print('\nSkipping {} chapter {}:'.format(book, chapter))
                continue

    #dataFrame.to_csv('bible.csv')
    print('\nData saved as CSV.')
    return dataFrame

def writeToFile(frame, targetFilename, testFilename):
    print('Writing CSV to text files {}, {}'.format(targetFilename, testFilename))
    targetFile = open(targetFilename, 'w+')
    testFile   = open(testFilename, 'w+')
    for idx, row in frame.iterrows():
        targetFile.write(str(row['engVerse']) + '\n')
        testFile.write(  str(row['ipqVerse']) + '\n')
    print('Text files written, {} lines.'.format(idx))
    targetFile.close()
    testFile.close()

if __name__ == '__main__':
    frame = scrapeBible()
    writeToFile(frame, 'bible_eng.txt', 'bible_ipq.txt')




