ML_NLP_OKCupid
==============================

Using NLP to analyze OkCupid profile texts for my class 

--------

# Research Question: Predict sex with user profile text (essay0: about me) on Dating App OKCupid 
## Scripts 

1. **BERT_Topic_Modeling**: Read in raw dataset, Data Processing: Remove stopwords, run BERTopic to create matrix with topic probabilities and dataset with highest probable topic per profile text,
2. **BERTopic predict sex** Use topic probability matrix to predict sex with neural net (pytorch)
3. **BERT Text Classifier**: Use AutoModelSequenceClassification from hugging face to predict sex, fine-tune model in 3 epochs, run inference, evaluate  

## Dataset
The Dataset okcupid_profiles.csv comes from Kaggle and can be directly downloaded [here](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles). 

**Demographic Variables are:**  
* age
* status
* sex
* orientation
* body type
* diet
* drinks
* drugs
* education
* ethnicity
* height
* income
* job
* last_online
* location
* offspring
* pets
* religion
* sign
* smokes
* speaks


**Essay Questions are:**
* essay0: About Me / Self summary
* essay1: Current Goals / Aspirations
* essay2: My Golden Rule / My traits
* essay3: I could probably beat you at / Talent
* essay4: The last show I binged / Hobbies
* essay5: A perfect day / Moments
* essay6: I value / Needs
* essay7: The most private thing I'm willing to admit / Secrets
* essay8: What I'm actually looking for / Dating
