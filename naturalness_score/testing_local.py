import time
import torch
from classifier.model import Model, get_model

model = get_model()
st1 = time.time() 
text1 = "1,Q42,[MASK] Adams,[MASK] writer and [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
text_mask_fill = model.fill_mask(text1)
et1 = time.time()

st2 = time.time() 
text2 = "1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
perplexity = model.perplexity(text2)
et2 = time.time()

print('Fill Mask')
print('Execution time :', et1-st1, 'seconds')
print('Avarage Perplexity')
print('Execution time:', et2-st2, 'seconds')

# print(9831050005000007%6)

# Run In pipenv terminal 
# python naturalness_score/testing_local.py




# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# http POST http://127.0.0.1:8000/perplexity text="1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
