import urllib.request
page = urllib.request.urlopen('http://services.runescape.com/m=hiscore/ranking?table=0&category_type=0&time_filter=0&date=1519066080774&user=zezima')
print(page.read())
#Source: https://stackoverflow.com/questions/1843422/get-webpage-contents-with-python
