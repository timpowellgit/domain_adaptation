import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import re
import csv

topics = [file.split('.')[0] for file in os.listdir('../data/test_evaluation_task2a/STS.answers-forums') if 'stackexchange' in file]
print topics

driver = webdriver.PhantomJS()
with open('../data/unlabeled.txt', 'wb') as f:
	for topic in topics:

		for pagenumer in range(10):
			print topic, pagenumer
			driver.get('http://%s.stackexchange.com/questions?page=%s&sort=frequent' %(topic, pagenumer+1))

			questions = driver.find_elements_by_xpath('//div[@class="question-summary"]')
			summaries = [q.find_element_by_xpath('.//a[@class="question-hyperlink"]').get_attribute('href') for q in questions]
			for summary in summaries:
				driver.get(summary)
				# _ = WebDriverWait(driver2, 1).until(
	   #              EC.presence_of_element_located((By.CLASS_NAME, "footerwrap")) 
	   #          )
	            
				answers = driver.find_elements_by_xpath('//div[@class="answer"]')
				text = []
				for answer in answers:

					secondpara = answer.find_elements_by_xpath('.//p')

					if len(secondpara) > 1:
						text.append(secondpara[1].text.split('.')[0])
					elif len(secondpara) ==1:
						text.append(secondpara[0].text.split('.')[0])


				text = [x for x in text if len(x)> 6]
				if len(text) %2 !=0:
					text.remove(text[-1])
				if len(text) >1:
					count=0
					while count < len(text)-1:

						towrite = text[count], '\t', text[count+1]
						print >> f, text[count].encode('utf8'), '\t', text[count+1].encode('utf8')
						count +=1
			driver.back()
