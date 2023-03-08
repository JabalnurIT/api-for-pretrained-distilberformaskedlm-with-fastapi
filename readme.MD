## Quick Start Guide

1. Clone the project from this repository:

```bash
git clone https://github.com/JabalnurIT/api-for-pretrained-distilbertmodelformaskedlm-with-fastapi.git
```

2. Open a new bash terminal (it is recommended to use bash) and install the dependency

```bash
pipenv install --dev
```

After finish the installation, run the shell pipenv (In case you don't have pipenv, try to google how to install it)

```bash
pipenv shell
```

That bash terminal should change to pipenv terminal. After that run this code on that terminal.

```bash
chmod +x bin/start_server
bin/start_server
```

3. Download the pretrained model from this link:

```bash
chmod +x bin/download_model
bin/download_model
```

After that, put the file inside the project assets.

```bash
./assets/
```

4. Open a new bash terminal and run this into the terminal, each code for one function, make sure you have install httpie on your computer

```bash
http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] [MASK],[MASK] [MASK] and [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"

http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK],british singer and writer,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
```

In the code above to use the fill_mask_text function. Don't forget to put [MASK] on it.

```bash
http POST http://127.0.0.1:8000/perplexity text="1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
http POST http://127.0.0.1:8000/perplexity text="1,Q42,q [MASK],british singer  and writer,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
```

In the code above to use the perplexity function. Don't put [MASK] on it.

## Support

Found this repo usefull? Support this effort by giving a star on GitHub, and don't be shy to collaborate with me. Meet me in [instagram](https://www.instagram.com/jibi.bm2/).
