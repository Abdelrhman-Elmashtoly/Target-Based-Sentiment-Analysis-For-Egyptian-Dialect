import numpy as np
from flask import Flask, request, render_template
import pickle 
import joblib
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification,BertForSequenceClassification, AutoModelForMaskedLM,BertForPreTraining 
from arabert.preprocess import ArabertPreprocessor
from arabert import ArabertPreprocessor
import torch
import re
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import string
from plotly.offline import init_notebook_mode, iplot
#import tensorflow as tf
import tensorflow
#from keras_preprocessing.sequence import pad_sequences
#from keras_preprocessing.sequence import pad_sequences
import nltk
#from flask.ext.assets import Environment, Bundle
#from flask.ext.scss import Scss

from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# model_name2 = "aubmindlab/bert-base-arabertv2"
# arabert_prep = ArabertPreprocessor(model_name=model_name2)
# model_ = AutoModel.from_pretrained(model_name2)
# model_.eval()
# tokenizer_2 = AutoTokenizer.from_pretrained(model_name2)
model_name="aubmindlab/bert-base-arabertv02-twitter"
arabert_prep = ArabertPreprocessor(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
PATH="model.pth"

model_d = torch.load(PATH,map_location =torch.device('cpu'))

print(model_d)


def data_cleaning (text):
    
    #text = text.strip()

    text= re.sub(r'http\S+', '', text)
    
    text= re.sub(r'ي+','ي', text)
    text = text.replace("آ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("أ", "ا")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")


    text= re.sub(r'[@|#]\S*', '',text)
    text= re.sub(r'"+', '', text)
    # Remove arabic signs
    #text= re.sub(r'([@A-Za-z0-9_ـــــــــــــ]+)|[^\w\s]|#|http\S+', '', text)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
   
    

    # Remove repeated letters like "الللللللللللللللله" to "الله"
    text= text[0:2] + ''.join([text[i] for i in range(2, len(text)) if text[i]!=text[i-1] or text[i]!=text[i-2]])
    text= re.sub(r'D',':D', text)
    text= re.sub(r'هه+', 'face_with_tears_of_joy', text)
    #text= convert_emojis(text)
    #text= convert_emoticons(text)

     # Removing punctuations in string
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text=re.sub(r'(?:^| )\w(?:$| )', ' ', text)
    text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", text)
    text = re.sub(" \d+", " ", text)
    text = re.sub("(\s\d+)","",text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub("(\s\d+)","",text) 
    # Returns: hi what is the weather like
    

    return text
import fasttext.util
fasttext.util.download_model('arz',if_exists='ignore')
Egyara_model=fasttext.load_model('cc.arz.300.bin')
from transformers import pipeline
import stanza
import spacy_stanza
stanza.download('ar')
# from transformers import AutoTokenizer, AutoModel
# from arabert.preprocess import ArabertPreprocessor

pos = pipeline('token-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-msa-pos-egy',grouped_entities=True, ignore_subwords=True,)

ar_stopwords = '''
أنفسنا مثل حيث ذلك بشكل لدى ألا عن إلي لنا وقالت فقط الذي الذى هذا غير أكثر اي أنا أنت ايضا اذا كيف وكل أو اكثر أي أن منه وكان وفي تلك إن سوف حين نفسها هكذا قبل حول منذ هنا عندما على ضمن لكن فيه عليه قليل صباحا لهم بان يكون بأن أما هناك مع فوق بسبب ما لا هذه  فيها ولم  آخر ثانية انه من الان دا به بن بعض حاليا بها هم كانت هي لها نحن تم أنفسهم ينبغي إلى فان وقد تحت' عند وجود الى فأن الي او قد خارج إنه اى مرة هؤلاء أنها إذا فهي فهى كل يمكن جميع أنفسكم فعل كان ثم لي الآن وقال فى في ديك لم لن له تكون الذين ليس التى التي أنه وان بعد حتى ان دون وأن لماذا يجري كلا إنها لك ضد وإن فهو انها منها أى لديه ولا بين خلال وما اما عليها بعيدا كما نفسي نحو هو نفسك نفسه انت ولن امبارح بصراحة صراحة النسبه عشان إضافي لقاء وكانت هى فما أيضا إلا معظم ومن إما الا بينما وهي وهو وهى
'''
ar_stopwords=nltk.word_tokenize(ar_stopwords)

def process_text(text):
    stemmer = nltk.ISRIStemmer()
    word_list = nltk.word_tokenize(text)
    #remove arabic stopwords
    word_list = [ w for w in word_list if not w in ar_stopwords ]
    #remove digits
    word_list = [ w for w in word_list if not w.isdigit() ]
    #stemming
    word_list = [stemmer.stem(w) for w in  word_list]
    #return ' '.join(word_list) 
    return text

def clean_text(text):
    search = ["أ","إ","آ","_","-","/",".","،"," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']  
    #word_list = [stemmer.stem(w) for w in  word_list]
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel,"", text)
  
    longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(longation, subst, text)
    
    text = re.sub(r"[^\w\s]", '', text)
    #Remove reputation
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    stemmer = nltk.ISRIStemmer()
    word_list = nltk.word_tokenize(text)
    word_list = [ w for w in word_list if not w in ar_stopwords ]   
    text = text.strip()
    return ' '.join(word_list) 
    #return text


def get_target_token(text):
    text_preprocessed=arabert_prep.preprocess(text)
    inputs = tokenizer.encode_plus(text_preprocessed, return_tensors='pt')
    input_ids=inputs['input_ids'][0]
    tokens=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    return    tokens 

def clean_tok_list(toks):
  cleantx=[]
  lt =len(toks)-1
  sent=toks[1:lt]

  
  c=[]
  for tokt in sent:
    ts=[]
    if tokt==sent[-1]:

      if not (re.search(r'##',tokt)) :
        c.append(tokt)
    else:
      if re.search(r'##',sent[sent.index(tokt)+1]):
        
        ts.append(tokt)
        tokt2=sent[sent.index(tokt)+1].replace('##', '')
        ts.append(tokt2)
        ts=''.join(ts)
        if re.search(r'##',ts):
          z=[]
          tokt3=c[-1]
          l = len(c)-1
          c=c[0:l]
          z.append(tokt3)
          z.append(tokt2) 
          ts=''.join(z)

        c.append(ts)


      elif not (re.search(r'##',tokt)) :
        c.append(tokt)

  for tokt in c:
      if tokt in Egyara_model.get_words() :
        cleantx.append(tokt)
      else:
        pt=pos(tokt)
        for i in range(len(pt)):
          if  pt[i]['entity_group']=='conj':
            cleantx.append(pt[i]['word'])

  return ' '.join(cleantx)

def clean__ (cleand_pos_text):
    
    for i in range(len(cleand_pos_text)):
        if len(cleand_pos_text[i]['word'].split())>1:
          list_T= cleand_pos_text[i]['word'].split() 

          list_T_POS= pos(list_T)
          cleand_pos_text.remove(cleand_pos_text[i])
          for j , k in zip( range(i, i+len(list_T_POS)) , range(len(list_T_POS))):
            cleand_pos_text.insert(j, list_T_POS[k][0])
                
    return cleand_pos_text 

nlp = spacy_stanza.load_pipeline('ar',
                     processors = 'tokenize,mwt,pos,lemma,depparse')

from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
from spacy import displacy

def get_mutible_sentences(text_t):

    #text=" القميص شكل وخامه محترم جدا وبالنسبه للسعر حلو  "
    text_t = text_t.strip()
    text=clean_text(text_t)
    doc = nlp(text)
    sentence_spans = list(doc.sents)
    
    text_ =[]
    dep_ =[]
    pos_=[]
    head=[]
    childs =[]
    for token in doc:

        text_.append(str(token.text))
        dep_.append( str(token.dep_))
        pos_.append(str(token.pos_ ))
        head.append(str(token.head.text))
        childs.append(str([child for child in token.children]))
        #output.append(t)  
    d = {'Token': text_, 'Relation': dep_, 'pos': pos_, 'Head': head, 'Children': childs}

    output = pd.concat([pd.Series(v, name=k) for k, v in d.items()], axis=1)

    #display(output)
    
    
    a=[]
    b=[]
    for i in range(len(childs)):
        for t in output['Children'][i]:
            if len(output['Children'][i])==2  :
                b=[]
            else:    
                if t.isalpha() or not (t.isspace):
                    a.append(t)
                elif t ==',' or t ==']':
                   b.append(''.join(a)) 
                   a=[]
                    
        output['Children'][i]=b 
        b=[]
        
    output['pos_camel']=output['pos']
    sentences=[]
    try:
        
        for i in range(len(text_)) :
            t_cleaned=clean_text(text_[i])
            tokn=get_target_token(t_cleaned)
            cleand_tok=clean_tok_list(tokn)
            pp= pos(cleand_tok)
            p=clean__(pp)
            print (p[0]['entity_group'])
            output['pos_camel'][i]=p[0]['entity_group']
            #condition 
            
            if output['pos_camel'][i] == 'noun'  and len(output['Children'][i])==0:
                    q=0
                    for li in range(len(sentences)):
                        if output['Token'][i] in sentences[li]:
                            q=1
                    
                    if q==0:        
                        Children_N=[]
                                

                        Children_N.append(output['Token'][i])
                        #print(Children_N)
                        sentences.append(' '.join(Children_N))
                    
            elif output['pos_camel'][i] == 'noun'  and len(output['Children'][i])!=0:
                    Children_N=[]


                    for chil in output['Children'][i]:
                        chil_cleaned=clean_text(chil)
                        tokchil=get_target_token(chil_cleaned)
                        cleand_chil=clean_tok_list(tokchil)
                        p_chil_=pos(cleand_chil)
                        p_chil=clean__(p_chil_)
                        print(p_chil)
                        # if   p_chil[0]['entity_group'] != 'noun'  and p_chil[0]['entity_group'] != 'adj' and p_chil[0]['entity_group'] != 'noun_prop' :
                        #     Children_N=[]
                                    
                        #     print(output['Token'][i])
                        #     Children_N.append(output['Token'][i])
                        #     #print(Children_N)
                        #     sentences.append(' '.join(Children_N))

                        if p_chil[0]['entity_group']== 'adj':
                            indx0=np.where(output.Token.str.contains(chil))[0].tolist()
                            print(indx0[0])
                            for r in indx0 :
                                print(r)
                                if output.Head[r] == output['Token'][i] :
                                    #print(output['pos_camel'][r-1])
                                    part_cleaned=clean_text(output['Token'][r-1])
                                    tokpart=get_target_token(part_cleaned)
                                    cleand_part=clean_tok_list(tokpart)
                                    p_part_=pos(cleand_part)
                                    p_part=clean__(p_part_)
                                    print(p_part[0]['entity_group'])
                                    
                                    if r!=0 and p_part[0]['entity_group'] == 'part_neg':
                                        
                                        Children_N.append(output['Token'][i])
                                        Children_N.append(output['Token'][r-1])
                                        Children_N.append(chil)
                                        sentences.append(' '.join(Children_N)) 
                                        Children_N=[]
                                    else:     
                                        
                                        Children_N.append(output['Token'][i])
                                        Children_N.append(chil)
                                        sentences.append(' '.join(Children_N)) 
                                        Children_N=[]

                        elif p_chil[0]['entity_group']== 'noun' :
                            indx0=np.where(output.Token.str.contains(chil))[0].tolist()
                            for r in indx0 :
                                if r > i : 
                                    c_cleaned=clean_text(output['Token'][i+1])
                                    tokc=get_target_token(c_cleaned)
                                    cleand_c=clean_tok_list(tokc)

                                    print(cleand_c)
                                    if cleand_c == p_chil[0]['word']:
                                        Children_N=[]
                                        Children_N.append(output['Token'][i])
                                        Children_N.append(output['Token'][i+1])


                                    sentences.append(' '.join(Children_N))
                                    ####
                                    #Children_N.append(chil)
                                    #Children_N.append(output['Token'][i])
                                    #sentences.append(' '.join(Children_N)) 
                                    #Children_N=[]
                                    indx=np.where(output.Token.str.contains(chil))[0].tolist()
                                    for k in indx:
                                        if output.Head[k] == output['Token'][i] :
                                            if len(output['Children'][k])==0:
                                                Children_N.append(output['Token'][i])
                                                Children_N.append(chil)
                                                sentences.append(' '.join(Children_N)) 
                                                Children_N=[]
                                            else:    


                                                    for c in output['Children'][i]:
                                                        c_cleaned=clean_text(c)
                                                        tokc=get_target_token(c_cleaned)
                                                        cleand_c=clean_tok_list(tokc)
                                                        p_c_=pos(cleand_c)
                                                        p_c=clean__(p_c_)
                                                        if p_c[0]['entity_group']== 'noun'  and p_chil[0]['word'] == p_c[0]['word']:
                                                            indx=np.where(output.Token.str.contains(c))[0].tolist()
                                                            for k in indx:
                                                                if output.Head[k] == output['Token'][i] :
                                                                    Children_N.append(output['Token'][i])
                                                                    if len(output['Children'][k])!=0:
                                                                        for l in output['Children'][k]:
                                                                            l_cleaned=clean_text(l)
                                                                            tokl=get_target_token(l_cleaned)
                                                                            cleand_l=clean_tok_list(tokl)
                                                                            p_l_=pos(cleand_l)
                                                                            p_l=clean__(p_l_)
                                                                            if p_l[0]['entity_group']== 'noun' or p_l[0]['entity_group']== 'noun_prop':
                                                                                l_c=clean_text(output['Token'][k+1])
                                                                                t_l=get_target_token(l_c)
                                                                                c_Lc=clean_tok_list(t_l)
                                                                                if c_Lc == p_l[0]['word']:


                                                                                    Children_N.append(output['Token'][k+1])

                                                    Children_N.append(chil)
                                                    sentences.append(' '.join(Children_N)) 
                                                    Children_N=[]



                #print(Children_N)
                #sentences.append(' '.join(Children_N))


            elif output['pos_camel'][i] == 'adj' and len(output['Children'][i])==0:
                    q=0
                    for li in range(len(sentences)):
                        if output['Token'][i] in sentences[li]:
                            q=1
                    
                    if q==0: 
                        if output['pos_camel'][i-1] == 'part_neg':
                            Children_N=[]
                                
                            Children_N.append(output['Token'][i-1])
                            Children_N.append(output['Token'][i])
                            print(Children_N)
                            sentences.append(' '.join(Children_N))
                        else:
                            Children_N=[]


                            Children_N.append(output['Token'][i])
                            print(Children_N)
                            sentences.append(' '.join(Children_N))
                    
            elif output['pos_camel'][i] == 'adj' and len(output['Children'][i])!=0:
                Children_N=[]

                
                    
                for chil in output['Children'][i]:
                    chil_cleaned=clean_text(chil)
                    tokchil=get_target_token(chil_cleaned)
                    cleand_chil=clean_tok_list(tokchil)
                    p_chil_=pos(cleand_chil)
                    p_chil=clean__(p_chil_)
                    #print(p_chil)
                    if p_chil[0]['entity_group']== 'noun'  :

                        Children_N.append(chil)
                        if output['pos_camel'][i-1] == 'part_neg':
                            Children_N.append(output['Token'][i-1])
                        #Children_N.append(output['Token'][i])
                        #sentences.append(' '.join(Children_N)) 
                        #Children_N=[]
                        indx=np.where(output.Token.str.contains(chil))[0].tolist()
                        for k in indx:
                            if output.Head[k] == output['Token'][i] :
                                if len(output['Children'][k])!=0:
                                    

                                        
                                    for l in output['Children'][k]:
                                        l_cleaned=clean_text(l)
                                        tokl=get_target_token(l_cleaned)
                                        cleand_l=clean_tok_list(tokl)
                                        p_l_=pos(cleand_l)
                                        p_l=clean__(p_l_)
                                        if p_l[0]['entity_group']== 'noun' or p_l[0]['entity_group']== 'noun_prop':
                                            indx1=np.where(output.Token.str.contains(l))[0].tolist()
                                            for u in indx1: 
                                                if u > k :
                                                    l_c=clean_text(output['Token'][k+1])
                                                    t_l=get_target_token(l_c)
                                                    c_Lc=clean_tok_list(t_l)
                                                    if c_Lc == p_l[0]['word']:


                                                        Children_N.append(output['Token'][k+1])

                        Children_N.append(output['Token'][i])
                        sentences.append(' '.join(Children_N)) 
                        Children_N=[]                             


                    elif p_chil[0]['entity_group']== 'adj'  :
                        #c= output['Children'][i]
                        indx7=np.where(output.Token.str.contains(chil))[0].tolist()
                        for k in indx7:
                            print(k)
                            if output.Head[k] == output['Token'][i] :
                              if len(output.Children[k])!=0:
                                c= output.Children[k]
                                print(c)
                                c_P=pos(c)
                                tp = []
                                for o in range(len(c_P)):
                                    tp.append(c_P[o][0]["entity_group"])
                                    
                                if 'noun' not in tp: 
                                  

                                  for c in output['Children'][i]:
                                    
                                      c_cleaned=clean_text(c)
                                      tokc=get_target_token(c_cleaned)
                                      cleand_c=clean_tok_list(tokc)
                                      p_c_=pos(cleand_c)
                                      p_c=clean__(p_c_)
                                      if p_c[0]['entity_group']== 'noun'  :
                                          
                                          Children_N=[]
                                          Children_N.append(c)

                                          Children_N.append(chil)
                                  sentences.append(' '.join(Children_N))
                                  Children_N=[]
                              else :

                                    for c in output['Children'][i]:
                                    
                                      c_cleaned=clean_text(c)
                                      tokc=get_target_token(c_cleaned)
                                      cleand_c=clean_tok_list(tokc)
                                      p_c_=pos(cleand_c)
                                      p_c=clean__(p_c_)
                                      if p_c[0]['entity_group']== 'noun'  :
                                          
                                          Children_N=[]
                                          Children_N.append(c)

                                          Children_N.append(chil)
                                            
                                    sentences.append(' '.join(Children_N))
                                    Children_N=[]
                            #print(p_chil)
                        #Children_N.append(chil)

                #Children_N.append(output['Token'][i])
                #print(Children_N)
              # sentences.append(' '.join(Children_N))
    except Exception as e:
              sentences.append(text_t)    
    
    #display(output)    
    return sentences , output

def get_targets(sentences):
    output_target = []
    
    for s in sentences:
        out=[]
        if len(s)!=0:
            
            pos_ =pos(s)
            pos_s=clean__(pos_)
            for i in range(len(pos_s)):
                
                if pos_s[i]['entity_group']=='noun' or pos_s[i]['entity_group']=='noun_prop' :
                    #if pos_s[i]['word']!='عشان' and pos_s[i]['word']!='النسبه':
                        if pos_s[i]['word'] not in out and len(pos_s[i]['word'])!=1: 
                            out.append(pos_s[i]['word'])
            
            y = ' '.join(out)
            if y not in output_target and y!='جدا' and y!='عشان' and y!='النسبه' and y!='': 
                  output_target.append(y)   
    
    
    return  output_target   

def clean_sentences(sentences):
    output=[]
    for s in sentences:
        if len(s)!=0:
            
            pos_ =pos(s)
            pos_s=clean__(pos_)
            tp = []
            for i in range(len(pos_s)):
                tp.append(pos_s[i]["entity_group"])
                
            if 'noun' in tp and 'adj' in tp and 'جدا' not in s and 'النسبه' not in s:
                ty=[]
                pos_ =pos(s)
                pos_s=clean__(pos_)
                for i in range(len(pos_s)):
                  
                  if len(pos_s[i]['word']) != 1:
                    ty.append(pos_s[i]['word'])
                output.append(' '.join(ty))  
                
    if len(output)==0:
        output = ' '.join(sentences)
            
    return output         
            
            
app= Flask(__name__, static_folder='static')

@app.route('/',methods=['GET'])
def hello():
   
    return render_template('landing-page-1-mobile-app.html',**locals())

#def homepage():
 #   return render_template('index.html',**locals())
labels=['positive','negative','neutral']
Encodedlabels =le.fit_transform(labels)
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
@app.route('/predict', methods = ['POST','GET'])
def predict():
    text=request.form['input_text']
    in_val=text
    #print(text)
    sentences, output=get_mutible_sentences(text)
    #print(sentences)
    #print('ddddddd')
    targets = get_targets(sentences)
    #print(targets)
    new_sentences = clean_sentences(sentences)
    result=new_sentences
    
    result2=targets

# shrief
    # def rslt2():
    #     for i in targets:
    #         print(i)
# shrief
    #df['text']=lis 
    #display(df)

    print(result)
    print("=========================================")
    df_submit = pd.DataFrame(result,columns=['tx'])
    #df_submit["Id"] = [0]
    #df_submit["tweet"] = [new_sentences]
    print(df_submit)
    # clean-up: remove #tags, http links and special symbols
    #df_submit['preprocessing'] =df_submit.iloc[:,0].apply(lambda x:data_cleaning(x))
    df_submit['preprocessing'] =df_submit.iloc[:,0] .apply(lambda x:arabert_prep.preprocess(x))

    df_submit["bert_tokens"] = df_submit.preprocessing.apply(lambda x: tokenizer(x).tokens())   
    df_submit["encoded"] = df_submit.tx.apply(lambda x: tokenizer.encode_plus(x,return_tensors='pt')['input_ids'])
    df_submit
    MAX_LEN = 256
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids_submit = [tokenizer.convert_tokens_to_ids(x) for x in df_submit["bert_tokens"] ]
    # Pad our input tokens
    input_ids_submit = pad_sequences(input_ids_submit, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks_submit = []

    
    for seq in input_ids_submit:
        seq_mask = [float(i>0) for i in seq]
        attention_masks_submit.append(seq_mask)
    inputs_submit = torch.tensor(input_ids_submit)
    masks_submit = torch.tensor(attention_masks_submit)

    
    batch_size =32
    submit_data = TensorDataset(inputs_submit, masks_submit)

    submit_dataloader = DataLoader(submit_data, batch_size=batch_size)#, shuffle=True)
    
    outputs = []
    for input, masks in submit_dataloader:
        #torch.cuda.empty_cache() # empty the gpu memory
        output = model_d(input, attention_mask=masks)["logits"]
        output = output.detach().numpy()

    # Store the output in a list
    outputs.append(output)

    # Concatenate all the lists within the list into one list
    outputs = [x for y in outputs for x in y]

    # Inverse transform the label encoding
    pred_flat = np.argmax(outputs, axis=1).flatten()
    output_labels = le.inverse_transform(pred_flat)
    submission = pd.DataFrame({"class":output_labels})
    result3=submission['class'].values
    print(submission.iloc[:,0])
    
    return render_template('landing-page-1-mobile-app.html',**locals())    
if __name__ == "__main__":
    app.run(debug=True,port=8000)
    
