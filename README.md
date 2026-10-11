install python in your machine 
pip install requirements.txt
 download ollama for both windows and mac link :https://ollama.com/download
 we used llama3 and nomic-embed-text
 after installing ollama
 open cmd  and type these commands:

 -  ollama serve    # this will start ollama service in 127.0.0.1:11434
 -  ollama pull nomic-embed-text   # this downloads  model from ollama server for sentence formation
 -  ollama pull llama3     # this downloads model from sentence for understanding and generating language
 -  streamlit run query_data.py     # open cmd in the folder and run this command to open a advanced rag chatbot according to the pdf given

you can add any pdf to the data directory and simply run :

 - python populate_database.py    #this will add the pdf and make it ready for chatting


there are pre inserted pdf for demo about monopoly and cybersecurity
you can test this to find the accuracy of the model

thank you..