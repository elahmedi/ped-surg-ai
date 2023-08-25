# ped-surg-ai
This repo is composed of two scripts:
1. qa.py: This is an LLM app that answers questions based on content from the 112 studies that were included in Elahmedi et al. systematic review on AI in pediatric surgery. It uses a support vector machine to select the paper that most closely matches the question. Then, the app feeds OpenAI's GPT 3.5 16k model the question and the paper, and the API returns the answer.

2. docsummary.py: This is an LLM app that generates a summary of all studies based on a theme of interest. It maps each study to an individual summary, then reduces the summaries to a single global summary. The current theme is equity, diversity and inclusivity. Due to the need for advanced reasoning capabilities, OpenAI's GPT-4 model is used here.
Steps:
1. Install the required dependencies (pip3 install openai langchain tiktoken sklearn)
2. Export your OpenAI API key.
3. Run the model

WARNING: Due to the large number of included studies and gpt-4's cost per token, running docsummary.py is costly.