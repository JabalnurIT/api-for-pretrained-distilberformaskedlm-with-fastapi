import time
import pandas as pd
from classifier.model import get_model

model = get_model()
# print(model[0])
# print(model.data_collator)
# st1 = time.time() 
# text1 = "Q42,[MASK] [MASK],[MASK] [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# text1 = "1, Q42,[MASK], [MASK] [MASK] [MASK],male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# text1 = "[MASK], [MASK],John [MASK],1st of the United States (1732–1799),[MASK],United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"
# text1 = "[MASK],[MASK],[MASK],[MASK],[MASK],[MASK],[MASK],[MASK],[MASK],[MASK],[MASK]"
# text_mask_fill = model.fill_mask(text1)
# print(text_mask_fill)
# et1 = time.time()


# =======================
# text2 = "1,Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"
# text2 = "2,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# text2 = "3,Q91,Abraham Lincoln,16th president of the United States (1809-1865),Male,United States of America,Politician,1809,1865.0,homicide,56.0"
# text2 = "4,Q254,Wolfgang Amadeus Mozart,Austrian composer of the Classical period,Male,Archduchy of Austria; Archbishopric of Salzburg,Artist,1756,1791.0,,35.0"
text2 = "5,Q42,Douglas Adams,English writer and 4t&$21,#34?sw,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# print(model.perplexity(text2))
# text2 = "Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"
text1 = text2
threshold = model.perplexity(text2)
# print(threshold)
lenght = len(text2.split(","))

for i in range(lenght):

    lenght2 = len(text2.split(",")[i].split(" "))
    for j in range(lenght2):
        text3 = text2.split((","))
        temp = text3[i].split(" ")
        temp[j] = "[MASK]"
        text3[i] = temp
        text3[i] = " ".join(text3[i])
        text3 = ",".join(text3)
        print(text3)
        text3 = model.fill_mask(text3)

        # text2 = text3
        perplexity = model.perplexity(text3)
        print(threshold)
        if perplexity < threshold:
            threshold = (perplexity+threshold)/2
            text2 = text3
        print(text3)
        print(text2)
        print(perplexity)
        input()
        print("=======")
    print(text2)
    print("####")

# print(model.perplexity(text2))
# print(text2)
# print(threshold)
# print(text1)

# =======================

# for i, word in enumerate(text2):
#     word = word.split(" ")
#     for j, word2 in enumerate(word):
#        word[j] = "[MASK]"
#     text2[i] = word
#     print(text2)

# text2 = "0,0,0,0,0,0,0,0,0,."

# et2 = time.time()

# print('Fill Mask')
# print('Execution time :', et1-st1, 'seconds')
# print('Avarage Perplexity')
# print('Execution time:', et2-st2, 'seconds')


# text3 = ['Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0',
#  'Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0',
#  'Q91,Abraham Lincoln,16th president of the United States (1809-1865),Male,United States of America,Politician,1809,1865.0,homicide,56.0',
#  'Q254,Wolfgang Amadeus Mozart,Austrian composer of the Classical period,Male,Archduchy of Austria; Archbishopric of Salzburg,Artist,1756,1791.0,,35.0']
# text3 = pd.DataFrame(text3)
# print(text3)
# retrain = model.retrain(text3)

# Run In pipenv terminal 
# python naturalness_score/testing_local.py




# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# http POST http://127.0.0.1:8000/perplexity text="1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
