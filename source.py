from tkinter.tix import MAX, Tree
from unicodedata import bidirectional
from unittest.util import _Max_length
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from torchmetrics.text.rouge import ROUGEScore
## Pre Processing code ##


train_data1 = pd.read_csv('CNN_Daily_News_train.csv')
test_data1 = pd.read_csv('CNN_Daily_News_test.csv')
   
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
listl=[]
for i in train_data1:
    no_punct = ""
    for char in i:
        if char not in punctuations:
            no_punct = no_punct + char
       
   
    listl.append(no_punct)

listl1=[]
for i in test_data1:

    no_punct = ""
    for char in i:
        if char not in punctuations:
            no_punct = no_punct + char
       
   
    listl1.append(no_punct)


token_list = []
print("\n")

for i in listl:
    my_doc = stopwords(i)   
    for token in my_doc:
        token_list.append(token.text)
   

for i in token_list:
    if not i.strip():
        token_list.remove(i)


train_data=[]        
test_data=[]  


for rf in token_list:
     
     for row in rf:
        train_data.append(row[1])

for rf in token_list:
     reader = input(rf, delimiter=',')
     for row in reader:
        test_data.append(row[1])



vocab=[]
for i in range(len(test_data)):
    line1=test_data[i][0].split(' ')
    for j in line1:
        vocab.append(j)


for i in range(len(train_data)):
    line1=train_data[i][0].split(' ')
    for j in line1:
        vocab.append(j)

vocab=set(vocab)
vocab_size=len(vocab)
## Vocabulary inserting code

## Initalization
Go_Token=0
EOS_Token=1
Max_length= 1000 # Maximum length of the text need to be summarized
embedding_size=50
lamda=2

torch.manual_seed(1)


## Pretrained Model##
word2vec = {}
embedding_dim = 50
f = open('glove.6B/glove.6B.50d.txt')
for line in f:
    values = line.split()
    name = values[0]
    values = np.asarray(values[1:], dtype='float32')
    word2vec[name] = values
f.close()



## Model Architecture ##
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,vocab_size, n_layers=1, drop_prob=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True,bidirectional=True) # Encoder is a BiLSTM
        self.layer1=nn.Linear(self.hidden_size*2,self.vocab_size)
        
    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        # output is set of encoder hidden states
        # hidden is the last encoder hidden state
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size,n_layers=1, drop_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size # Dimension of hidden embedding
        self.n_layers = n_layers # Number of layers
        self.drop_prob = drop_prob # Drop out probability in the dropout layer

        self.embedding = nn.Embedding(self.embedding_size, self.hidden_size) #Embedding of a input words passed

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Linear layer added to generalize hidden state
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Linear layer added to generalize encoder state
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size)) # Weight matrix that is auto adjusted
        self.dropout = nn.Dropout(self.drop_prob) # Drop out layer added
        self.lstm = nn.LSTM(self.hidden_size*2,self.hidden_size, batch_first=True) # LSTM Layer added
        self.p_gen_fc=nn.Linear(self.hidden_size*4,1) # Linear layer to obtain Pgen from context vector,decoder hidden state and decoder input
        self.layer1=nn.Linear(self.hidden_size,vocab_size) # Layer to distribute lstm output in vocabulary
        self.coveragelayer=nn.Linear(self.hidden_size,self.hidden_size,bias=False) # Linear layer for coverage
   
    def forward(self, inputs, hidden, encoder_outputs,ext_vocab_size,coverage_vector):
        # inputs is the decoder input
        # hidden is the decoder hidden state - In the first time step it is last encoder hidden state
        # Encoder outputs is set of all encoder hidden states
        encoder_outputs = encoder_outputs.squeeze()
        # Embed input words
        embedded = self.embedding(inputs).view(1, -1)
        # Apply a drop out layer to avoid overfitting
        embedded = self.dropout(embedded)

        # Calculating Alignment Scores using decoder hidden state, encode hidden states,coverage vector
        # Formula is score(ht,hs1,ct)=vt.tanh.(wh.ht+ws.hs1+wc.ct+b)
        # where ht=decoder hidden state, hs1=encoder hidden states, ct=coverage vector and all these are at that time step
        # Remaining are learnable parameters
        x = torch.tanh(self.fc_hidden(hidden[0])+self.fc_encoder(encoder_outputs)+self.coveragelayer(self.coverage))
        alignment_scores = x.bmm(self.weight.unsqueeze(2))

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)
        normfactor=attn_weights.sum(1,keepdim=True)
        attn_weights=attn_weights/normfactor

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))


        # Input to decoder is the concatenator previous output of decoder and contextvector that has obtained
        decoder_input = torch.cat((embedded, context_vector[0]), 1).unsqueeze(0)

        #Passing this to LSTM layer
        output, decoder_hidden = self.lstm(decoder_input, hidden)

        #Additional layer that converts the output among the vocabulary          
        output=self.layer1(output)

        #Applying the softmax layer to obtain the probabilities
        prob_vocab = F.softmax(output, dim=1)
        
        coverage_vector=coverage_vector.view(-1,self.hidden_size)
        coverage_vector += attn_weights.view(-1,self.hidden_size)
        # Pgen generation
        p_gen_inp = torch.cat((context_vector, hidden[0], embedded), 1)  
        p_gen = self.p_gen_fc(p_gen_inp)
        p_gen = torch.sigmoid(p_gen)

        # Vocabulary distribution
        vocab_dist=p_gen*prob_vocab
        # Attention distribution in source text
        attn_dist=(1-p_gen)*attn_weights
        pad_dim = (0, ext_vocab_size - output.size(1))        

        # Pad the output tensir
        output=F.pad(output, pad_dim, 'constant')
        #This is the final extended vocabulary distribution
        final_dist=vocab_dist.scatter_add(1,attn_dist)

        return final_dist,coverage_vector,decoder_hidden,attn_weights


## Initializing model and parameters ##
loss_criterion = nn.CrossEntropyLoss()
encoder=EncoderLSTM(300,50,vocab_size,3,0.5)
decoder=DecoderLSTM(50,1,0.5)

# Adam optimizer for encoder and decoder
encoder_optimizer=optim.Adam(encoder.parameters(),lr=0.01)
decoder_optimizer=optim.Adam(decoder.parameters(),lr=0.01)


def train(input_tensor,target_tensor,encoder,decoder, encoder_optimizer,decoder_optimizer,Max_length,extended_vocab_size):
    encoder_hidden=encoder.init_hidden()    

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    
    encoder_outputs = torch.zeros(Max_length, encoder.hidden_size)

    loss=0 # Intial loss to be propogated is 0

    for i in range(input_length):
        encoder_output,encoder_hidden=encoder(input_tensor[i],encoder_hidden)
        encoder_outputs[i]=encoder_output[0,0]

    decoder_input=torch.tensor([[Go_Token]])    
    decoder_hidden=encoder_hidden


    # Teacher Forcing
    for i in range(target_length):        
        extnd_vocab_dist,coverage_vector,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_output,coverage_vector,extended_vocab_size)

        x,y=extnd_vocab_dist.topk(1)
        decoder_input=y.squeeze().detach() # Detaching as this doesnt require a gradient
        
        logloss = loss_criterion(extnd_vocab_dist, target_tensor[i])
        coverage_loss = torch.sum(torch.min(coverage_vector, decoder_attention)) 

        loss+=logloss+lamda*coverage_loss
        # We got output as end of sentence, so this is end of our prediction
        if(decoder_input.item()==EOS_Token):
            break 

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    # Returning the average loss
    return (loss.item()/target_length)


def CreateSentenceTensor(sentence):
    indices=[word2vec[word] for word in sentence.split(' ')]
    indices.append(EOS_Token)
    return torch.tensor(indices,dtype=torch.long).view(-1,1)

def eval(input_tensor,target_tensor,encoder,decoder,sentence,Max_length,ext_vocab_size):    

    input_tensor=CreateSentenceTensor(sentence)
    input_length=input_tensor.size()[0]
    encoder_hidden=encoder.init_hidden()
    encoder_outputs = torch.zeros(Max_length, encoder.hidden_size)

    for i in range(input_length):
        encoder_output,encoder_hidden=encoder(input_tensor[i],encoder_hidden)
        encoder_outputs[i]=encoder_output[0,0]

    decoder_input=torch.tensor([[Go_Token]])    
    decoder_hidden=encoder_hidden
    

    decoded_words=[]
    decoded_attentions= torch.zeros(Max_length, Max_length)

    for i in range(Max_length):                

        extnd_vocab_dist,coverage_vector,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs,ext_vocab_size)
        x,y=extnd_vocab_dist.topk(1)
        
        # We got output as end of sentence, so this is end of our prediction
        if(decoder_input.item()==EOS_Token):
            decoded_words.append('<EOS>')
            break 
        else:
            decoded_words.append(word2vec[decoder_input.item()])

        decoder_input=y.squeeze().detach() # Detaching as this doesnt require a gradient

    rouge=ROUGEScore()
    output_tensor=CreateSentenceTensor(decoded_words)
    scores = rouge.get_scores(output_tensor, target_tensor, True)

    rouge2_f_metric = scores['rouge-2']['f']
    rouge2_p_metric = scores['rouge-2']['p']
    rouge2_r_metric = scores['rouge-2']['r']
    rougel_f_metric = scores['rouge-l']['f']
    rougel_p_metric = scores['rouge-l']['p']
    rougel_r_metric = scores['rouge-l']['r']
    
    return decoded_words,scores
    

# Training Phase
for epochs in range(10):
    losses=[]
    for sentence in train_data:
        text=sentence[0]
        summary=sentence[1]
        vocab1=vocab
        tok=text.split(' ')
        for word in tok:
            vocab1.append(word)
        
        vocab1=set(vocab1)
        extended_vocab_size=vocab1
        input_tensor=CreateSentenceTensor(text)
        target_tensor=CreateSentenceTensor(summary)
        loss=train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,Max_length,extended_vocab_size)
        losses.append(loss)
    
    print("Current Epoch: {}".format(epochs))
    for i,loss in enumerate(losses):
        print("Loss for {} the sentence is {}".format(i,loss))    



# Testing
for sentence in test_data:
    text=sentence[0]
    summary=sentence[1]
    input_tensor=CreateSentenceTensor(text)
    target_tensor=CreateSentenceTensor(summary)
    vocab1=vocab
    tok=text.split(' ')
    for word in tok:
        vocab1.append(word)
    
    vocab1=set(vocab1)
    extended_vocab_size=vocab1
    decoded_words,scores=eval(input_tensor,target_tensor,encoder,decoder,sentence,Max_length,extended_vocab_size)

    decoded_sentence=' '.join(decoded_words)
    print("Summary for sentence is:\n")
    print(decoded_sentence)
    print("Rouge L scores: {}".format(scores))

# Saving the models
torch.save(encoder.state_dict(),'encoder.pt')
torch.save(decoder.state_dict(),'decoder.pt')
