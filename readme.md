## this project is created on 5 nov 2025. 'detoxify + nlp' approach did not work as expected. now i am moving to contextual_bias approach.
- i will have to deliver this project by tuesday 11-nov.
- set temp to 0 and seed fix 0. so answer will most likely remain same. 

## **installation guide:**
- create env: python3.10 -m venv venv
- activate env: source venv/bin/activate
- install library: pip install -r requirements.txt
---

## **TO DO LIST**
- first ask tl what spacy library version i should install. spacy earlier gave lots of issue.
# ======= Load SpaCy NLP model safely =======
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    logging.info("Installing SpaCy model en_core_web_sm ...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    ], check=True)
    import spacy
    nlp = spacy.load("en_core_web_sm")

requiremnets.txt: spacy==3.8.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl


- install library for milestone2_contextual_bias_analysis.py 
- run milestone2_contextual_bias_analysis.py on dataset_response.csv
- keep cuda so in future client might use it to speed up. client most likely is using Apple macbook m3.
---

## **core files**
- dataset/ dataset.json: new data which has 2 option for each prompt. new created data is based on race, gender, intersectional.

- dataset_response/ dataset_response.csv: response created by milestone1_gen_output.py. on this data i will run my   milestone2_contextual_bias_testing.py

- extra files/ : extra files like client msg, client image, bad dataset response, milestone, response to client.

- milestone1_gen_output.py: it will generate response from 3 llm and create new file from it in folder dataset_response.
- milestone2_contextual_bias_testing.py: it will work on file 'dataset_response.csv' that will be generated from milestone1_gen_output.py to give contextual_bias.

- milestone.md: milestone that i was suppose to follow.
- client_msg_5nov.md: client give this msg on 5 nov with a client_image.jpeg file
- client_image.jpeg:  image that client gave with client_msg_5nov.md 
- project_Structure.md: this is the structure i will follow for this project.