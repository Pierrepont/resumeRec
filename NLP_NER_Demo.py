import spacy
import textract
import csv
import numpy as np
import pandas as pd
def main(resume_list, jobDesc_list, skill_list, filename):
    #Extract all skills from all pdfs
    skill_dict=extract(resume_list+jobDesc_list, skill_list)
    #Initialize score system
    score_dict={}
    #Initialize output array
    out=[]
    #Loop thru all job descriptions
    for jobDesc in jobDesc_list:
        #Initialize score for that pdf
        score_dict[jobDesc]=[]
        #loop thru all resumes
        for resume in resume_list:
            #Add the score(number of matches) to score array for that pdf
            score_dict[jobDesc].append(len([skill for skill in skill_dict[jobDesc] if skill in skill_dict[resume]]))
        #Get all indices for top scoring resumes
        ids=[val==max(score_dict[jobDesc]) for val in score_dict[jobDesc]]
        #Add to output: job description, all filenames of top scoring arrays, their score, max score
        out.append([jobDesc, ', '.join(list(np.array(resume_list)[ids])), np.array(score_dict[jobDesc])[ids][0], len(skill_dict[jobDesc])])
    #Convert to dataframe for column names
    out=pd.DataFrame(out, columns=['Job Description', 'Matched Resumes', 'Score', 'Max Score'])
    #Save as csv
    out.to_csv(filename, sep=',',index=False)
    return('Success! Check '+filename+" for matches!")
        

def extract(pdfList, skill_list):
    wordList=[]
    #Loop thru all pdfs
    for pdf in pdfList:
        #Use tesseract.js to extract words from pdf
        text = textract.process(pdf, method='tesseract', language='eng')
        #add the text to array, text in bytes, needs to be decoded into string for spacy module
        wordList.append(text.decode())
    #load spacy model
    nlp = spacy.load("en_core_web_sm")
    #initialize output array
    out=[]
    #go through each pdf
    pdf_dict={}
    for i,pdf in enumerate(wordList):
        pdf_dict[pdfList[i]]=[]
        #do NER tagging on it
        doc = nlp(pdf)
        #retrieve all tagged words and its tags
        for ent in doc.ents:
            #append word and its tags to output array
            pdf_dict[pdfList[i]].append(ent.text)
        #trimming out to only have words in input_list
        pdf_dict[pdfList[i]]=list(np.unique([word for word in pdf_dict[pdfList[i]] if word in skill_list]))
    print('Skills Extracted!')
    return pdf_dict
print(main(['resume.pdf', 'resume1.pdf'], ['jobDesc.pdf', 'jobDesc1.pdf', 'jobDesc2.pdf'],['Apple','Python', 'SQL', 'Java', 'Javascript'], 'output1.csv'))