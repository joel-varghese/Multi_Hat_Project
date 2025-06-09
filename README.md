The goal of this project was to develop a product to summarize a pdf file into two sections based on a role selected by the user. 
This role is set as a context that drives the theme of the summary. Each section is achieved by a different modelling approach.


Skill based keyword summary : For this section a graph neural network using graphsage is used to extract keywords from the document and predict the edges that these keywords may have. Wherever this edge matches the role selected by the user that particular set of skills is displayed as a possible skill a user in that role must have.

Role based text summary : For this section a RAG pipeline is utilized to summarize the text in the role of an occupation selected by the user. Initially the pdf document is uploaded by the user and the extracted text is sent to the LLM, which in this case is mistral. Then, an additional context is provided with a question that queries the LLM to summarize the key information a professional of the selected field must know about the document. This summarized text is then displayed back on the UI screen

In order to train the dataset the European Skills, Competences, Qualifications and Occupations (ESCO) is used to create a JSON based format of roles and the particular keywords attached to that role as shown below : 
