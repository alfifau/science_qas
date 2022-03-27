# pylint: disable=import-error
from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)

global question_padding, answers_padding, question_vector, answers_vector, candidate_answers, sims

# Main page
@app.route('/', methods=['GET','POST'])
def index():
    global question_padding, answers_padding, question_vector, answers_vector, candidate_answers, sims
    if request.method == 'GET':
        return render_template('index.html')
    else:
        pertanyaan = request.form['question']
        kandidat = request.form['candidate']
        detail = '<br></br> <a href="preprocessing" class="btn btn-default btn-sm" role="button">Detail Proses</a>'
        
        ### proses pencari jawaban ###
        import gensim
        import nltk
        nltk.download('punkt')
        
        # 1. preprocessing pake gensim
        question_preprocessing = gensim.utils.simple_preprocess(pertanyaan)
        candidate_answers = [c for c in nltk.sent_tokenize(kandidat)]
        answers_preprocessing = [gensim.utils.simple_preprocess(c) for c in nltk.sent_tokenize(kandidat)]

        # 2. filter kata yang gaada di vocab
        from gensim.models import word2vec
        model_w2v = word2vec.Word2Vec.load("model/word2vec_100_sg_hs_20_dataset_pad.model")
        
        question_filter = []
        for word in question_preprocessing:
            if word in model_w2v.wv.vocab:
                question_filter.append(word)
            
        answers_filter = []
        for c in answers_preprocessing:
            word_filter = []
            for word in c:
                if word in model_w2v.wv.vocab:
                    word_filter.append(word)
            if len(word_filter) != 0:
                answers_filter.append(word_filter) 

        # 3. padding
        maxquestion = 15
        maxanswer = 57
        
        # padding question
        question_padding = question_filter.copy()
        if (len(question_filter) < maxquestion):
            for pad in range(maxquestion - len(question_filter)):
                question_padding.append('<PAD>')

        # padding answer
        answers_padding = []
        for c in answers_filter:
            if (len(c) < maxanswer):
                for pad in range(maxanswer - len(c)):
                    c.append('<PAD>')
            answers_padding.append(c)

        # 4. mengubah jadi vektor (tampilan)
        # mengubah kata menjadi vektor  
        question_vector = []
        question_vector.extend([model_w2v[word] for word in question_padding])

        answers_vector = []
        for c in answers_padding:
            answers_vector.append([model_w2v[word] for word in c]) 
        
        # 5. mengubah jadi index (word2vec)

        # mengubah kata menjadi index yang ada pada vocab model
        question_index = []
        question_index.append([model_w2v.wv.vocab[word].index for word in question_padding])

        answers_index = []
        for c in answers_padding:
            answers_index.append([model_w2v.wv.vocab[word].index for word in c])
        
        class QAData():
            def __init__(self):
                self.model = word2vec.Word2Vec.load('model/word2vec_100_sg_hs_20_dataset_pad.model')
                self.max_question = 15
                self.max_answer = 57

            def pad(self, data, length):
                """
                Pad data agar mempunyai panjang sesuai dengan panjang yang ditetapkan

                Args:
                    data (vector): vektor dari pertanyaan dan jawaban
                    length(integer): panjang dari vektor
                """

                from keras.preprocessing.sequence import pad_sequences
                # pad_sequence berarti yang masuk kesini harus sebuah sequence
                # padding = 'post' berarti yang dipadding setelah datanya
                # truncating = 'post' berarti menghapus sequence yg lebih panjang dari maxlen, dari yang belakang
                # value = 0 berarti selain data diganti dengan 0
                return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)
            
            def process_test_data(self, question, answers):
                """
                Proses data yang akan ditest
                Menduplicate pertanyaan untuk setiap kandidat jawaban
                """
                
                answers = self.pad(answers, self.max_answer)
                question = self.pad(question * len(answers), self.max_question)
                return question, answers

        # load training data
        qa_data = QAData()
        question_data, answers_data = qa_data.process_test_data(question_index, answers_index)
  
        from keras import backend as K
        from keras.layers import Embedding
        from keras.layers import LSTM, Input, merge, Lambda
        from keras.layers.wrappers import Bidirectional
        from keras.models import Model
        from keras.optimizers import Adam
        import numpy as np
        from gensim.models import word2vec
        
        class QAModel():
            def get_cosine_similarity(self):
                # lambda adalah suatu fungsi
                # 
                dot = lambda a, b: K.batch_dot(a, b, axes=1)
                return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())

            def get_bilstm_model(self):
                """
                Return the bilstm training and prediction model

                Args:
                    embedding_file (str): embedding file name
                    vacab_size (integer): size of the vocabulary

                Returns:
                    training_model: model used to train using cosine similarity loss
                    prediction_model: model used to predict the similarity
                """

                margin = 0.1
                max_question = 15 # maksimun question length
                max_answer = 57
                hidden_dim = 50

                # initialize the question and answer shapes and datatype
                question = Input(shape=(max_question,), dtype='int32', name='question_base')
                answer = Input(shape=(max_answer,), dtype='int32', name='answer')
                answer_good = Input(shape=(max_answer,), dtype='int32', name='answer_good_base')
                answer_bad = Input(shape=(max_answer,), dtype='int32', name='answer_bad_base')
                
                # load pre-trained word embedding
                model = word2vec.Word2Vec.load('model/word2vec_100_sg_hs_20_dataset_pad.model')
                embedding_matrix = np.zeros((len(model.wv.vocab), 100))
                for i in range(len(model.wv.vocab)):
                    embedding_vector = model.wv[model.wv.index2word[i]]
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        
                qa_embedding = Embedding(input_dim=embedding_matrix.shape[0],output_dim=embedding_matrix.shape[1],mask_zero=False,weights=[embedding_matrix], trainable=False)
                bi_lstm = Bidirectional(LSTM(units=hidden_dim, dropout=0.2, return_sequences=True))
                maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
            
                # embed the question and pass it through bilstm
                question_embedding =  qa_embedding(question)
                question_encoded = maxpool(bi_lstm(question_embedding))

                # embed the answer and pass it through bilstm
                answer_embedding =  qa_embedding(answer)
                answer_encoded = maxpool(bi_lstm(answer_embedding))

                # get the cosine similarity
                similarity = self.get_cosine_similarity()
                question_answer_similarity = Lambda(similarity)([question_encoded, answer_encoded])
                lstm_model = Model(name="bi_lstm", inputs=[question, answer], outputs=question_answer_similarity)
                good_similarity = lstm_model([question, answer_good])
                bad_similarity = lstm_model([question, answer_bad])

                # compute the loss
                loss = merge(
                    [good_similarity, bad_similarity],
                    mode=lambda x: K.relu(margin - x[0] + x[1]),
                    output_shape=lambda x: x[0]
                ) 
            
                # optimizer
                adam = Adam(lr=0.05)
            
                # return training and prediction model
                training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
                training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=adam, metrics=["acc"])
                prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
                prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=adam, metrics=["acc"])

                return training_model, prediction_model
        
        # load weights from trained model
        qa_model = QAModel()
        train_model, predict_model = qa_model.get_bilstm_model()
        predict_model.load_weights('model/train037_weights_epoch_100.h5')

        # mengambil data
        qa_data = QAData()
        question_data, answers_data = qa_data.process_test_data(question_index, answers_index)

        # mencari similarity score untuk tiap candidate
        sims = predict_model.predict([question_data, answers_data])
        jawaban = candidate_answers[np.argmax(sims)]

        for i in range(len(sims)):
            print(sims[i], candidate_answers[i])

        return render_template('index.html', pertanyaan=pertanyaan, kandidat=kandidat, answer=jawaban, detail=detail)


@app.route('/preprocessing')
def preprocessing():
    global question_padding, answers_padding
    # data = pickle.load(open('static/data/TrainData.pkl','rb'))
    return render_template('preprocessing.html', question=question_padding, answers=answers_padding)

@app.route('/vektor')
def vektor():
    global question_vector, answers_vector, question_padding, answers_padding
    return render_template('vektor.html', q_word = question_padding, a_word = answers_padding, question=question_vector, answers=answers_vector)

@app.route('/hasil')
def hasil():
    global sims, candidate_answers
    return render_template('hasil.html', answers=candidate_answers, similarity=sims)

if __name__ == '__main__':
    app.run()