import time
import pandas as pd
from classifier.model import get_model

model = get_model()
# print(model[0])
# print(model.data_collator)
# st1 = time.time() 
# text1 = "1,Q42,[MASK] [MASK],[MASK] [MASK] [MASK] [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# text_mask_fill = model.fill_mask(text1)
# print(text_mask_fill)
# et1 = time.time()

# st2 = time.time() 
# text2 = "1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# perplexity = model.perplexity(text2)
# et2 = time.time()

# print('Fill Mask')
# print('Execution time :', et1-st1, 'seconds')
# print('Avarage Perplexity')
# print('Execution time:', et2-st2, 'seconds')


text3 = ['Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0',
 'Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0',
 'Q91,Abraham Lincoln,16th president of the United States (1809-1865),Male,United States of America,Politician,1809,1865.0,homicide,56.0',
 'Q254,Wolfgang Amadeus Mozart,Austrian composer of the Classical period,Male,Archduchy of Austria; Archbishopric of Salzburg,Artist,1756,1791.0,,35.0']
# text3 = pd.DataFrame(text3)
# print(text3)
retrain = model.retrain(text3)

# Run In pipenv terminal 
# python naturalness_score/testing_local.py




# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# http POST http://127.0.0.1:8000/perplexity text="1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
