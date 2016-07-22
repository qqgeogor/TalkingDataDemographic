import pandas as pd
import numpy as np
import pylab as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.cross_validation import StratifiedKFold,KFold
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot

seed = 1024
max_index_label = 1021
maxlen = 500
dim = 128

path = "../input/"


def get_train_valid():
    train = pd.read_csv(path+'gender_age_train.csv')
    skf = StratifiedKFold(train['group'].values, n_folds=10, shuffle=True, random_state=seed)
    # skf = KFold(train.shape[0],n_folds=5, shuffle=True, random_state=seed)
    for ind_tr, ind_te in skf:
        X_train = train.iloc[ind_tr]
        X_test = train.iloc[ind_te]
        break

    X_train.to_csv(path+"X_train.csv",index=False)
    X_test.to_csv(path+"X_test.csv",index=False)

def get_events():
    label_categories = pd.read_csv(path+'label_categories.csv')

    events = pd.read_csv(path+'events.csv',
                    dtype={'device_id': np.str})

    app_events = pd.read_csv(path+'app_events.csv',
                    dtype={'device_id': np.str})

    app_labels = pd.read_csv(path+'app_labels.csv',
                    dtype={'device_id': np.str})


    ##################
    #   App Labels
    ##################
    print("# Read App Labels")
    app_labels = app_labels.groupby("app_id")["label_id"].apply(
        lambda x: " ".join(str(s) for s in x))

    ##################
    #   App Events
    ##################
    print("# Read App Events")
    # app_events = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
    app_events["app_labels"] = app_events["app_id"].map(app_labels)
    app_events = app_events.groupby("event_id")["app_labels"].apply(
        lambda x: " ".join(str(s) for s in x))
    
    del app_labels

    ##################
    #     Events
    ##################
    print("# Read Events")
    # events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
    events["app_labels"] = events["event_id"].map(app_events)
    events = events.groupby("device_id")["app_labels"].apply(
        lambda x: " ".join(str(s) for s in x))

    del app_events

    return events

def X_train_generatetor_infinite(dim=128,maxlen=500,batch_size=128,name="X_train.csv",events=None):
    X_train = pd.read_csv(path+name)
    group_le = LabelEncoder()
    group_lb = LabelBinarizer()
    labels = group_le.fit_transform(X_train['group'].values)
    labels = group_lb.fit_transform(labels)
    del labels
    
    ##################
    #   Phone Brand
    ##################
    # print("# Read Phone Brand")
    phone_brand_device_model = pd.read_csv(path+'phone_brand_device_model.csv',
                    dtype={'device_id': np.str})
    phone_brand_device_model.drop_duplicates('device_id', keep='first', inplace=True)
    phone_brand_le = LabelEncoder()
    phone_brand_device_model['phone_brand'] = phone_brand_le.fit_transform(phone_brand_device_model['phone_brand'])

    device_model_le = LabelEncoder()
    phone_brand_device_model['device_model'] = phone_brand_le.fit_transform(phone_brand_device_model['device_model'])


    while 1:
        data = pd.read_csv(path+name,iterator=True,chunksize=batch_size,
                    dtype={'device_id': np.str})
        for X_train in data:
            X_train = pd.merge(X_train,phone_brand_device_model,how='left',on='device_id', left_index=True)
            phone_brand = X_train['phone_brand'].values
            device_model = X_train['device_model'].values


            X_train["app_lab"] = X_train["device_id"].map(events)
            y_train = X_train['group'].values
            
            X_train['gender'][X_train['gender']=='M']=1
            X_train['gender'][X_train['gender']=='F']=0

            y_train_gender = X_train['gender'].values
            y_train_age = X_train['age'].values
            # take log transformation
            y_train_age = np.log(y_train_age)

            X_train.fillna('0 ',inplace=True)
            y_train = group_le.transform(y_train)
            y_train = group_lb.transform(y_train)
            x_train = X_train["app_lab"].values
            x_train = [ x.split(' ') for x in  x_train]
            for i in range(len(x_train)):
                x_train[i] = [ np.int8(idx) for idx in x_train[i] if (idx!='nan' and idx!='')]

            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            
            x_train = [x_train,phone_brand,device_model]
            y_train = [y_train,y_train_gender,y_train_age]

            yield (x_train,y_train)



def get_test(dim=128,maxlen=500,name='test.csv',events=None):
    X_train = pd.read_csv(path+name,
                    dtype={'device_id': np.str})
    X_train["app_lab"] = X_train["device_id"].map(events)
    X_train.fillna('0 ',inplace=True)
    x_train = X_train["app_lab"].values

    phone_brand_device_model = pd.read_csv(path+'phone_brand_device_model.csv',
                    dtype={'device_id': np.str})
    phone_brand_device_model.drop_duplicates('device_id', keep='first', inplace=True)

    phone_brand_le = LabelEncoder()
    phone_brand_device_model['phone_brand'] = phone_brand_le.fit_transform(phone_brand_device_model['phone_brand'])

    device_model_le = LabelEncoder()
    phone_brand_device_model['device_model'] = phone_brand_le.fit_transform(phone_brand_device_model['device_model'])


    X_train = pd.merge(X_train,phone_brand_device_model,how='left',on='device_id', left_index=True)
    X_train.fillna(0,inplace=True)
    phone_brand = X_train['phone_brand'].values
    device_model = X_train['device_model'].values

    x_train = [ x.split(' ') for x in  x_train]
    for i in range(len(x_train)):
        x_train[i] = [ np.int8(idx) for idx in x_train[i] if (idx!='nan' and idx!='')]

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_train = [x_train,phone_brand,device_model]
    return x_train

def main():
    phone_brand_device_model = pd.read_csv(path+'phone_brand_device_model.csv',
                    dtype={'device_id': np.str})
    num_p = len(np.unique(phone_brand_device_model['phone_brand']))
    num_d = len(np.unique(phone_brand_device_model['device_model']))
    
    del phone_brand_device_model
    
    X_train = pd.read_csv(path+'X_train.csv',
                    dtype={'device_id': np.str})
    X_test = pd.read_csv(path+'X_test.csv',
                    dtype={'device_id': np.str})

    inputs = Input(shape=(maxlen,), dtype='int32')
    
    embed = Embedding(
                    max_index_label,
                    dim,
                    dropout=0.2,
                    input_length=maxlen
                    )(inputs)


    inputs_p = Input(shape=(1,), dtype='int32')
    
    embed_p = Embedding(
                    num_p,
                    dim,
                    dropout=0.2,
                    input_length=1
                    )(inputs_p)

    inputs_d = Input(shape=(1,), dtype='int32')
    
    embed_d = Embedding(
                    num_d,
                    dim,
                    dropout=0.2,
                    input_length=1
                    )(inputs_d)

    conv_1 = Convolution1D(nb_filter=256,
                        filter_length=3,
                        border_mode='same',
                        activation='relu',
                        subsample_length=1)(embed)
    pool_1 = MaxPooling1D(pool_length=maxlen/2)(conv_1)



    flatten= Flatten()(pool_1)

    flatten_p= Flatten()(embed_p)

    flatten_d= Flatten()(embed_d)

    flatten = merge([flatten,flatten_p,flatten_d],mode='concat')
    
    fc1 = Dense(512)(flatten)
    fc1 = SReLU()(fc1)
    dp1 = Dropout(0.5)(fc1)
    
    fc2 = Dense(128)(dp1)
    fc2 = SReLU()(fc2)
    dp2 = Dropout(0.5)(fc2)


    fc1_g = Dense(512)(flatten)
    fc1_g = SReLU()(fc1_g)
    dp1_g = Dropout(0.5)(fc1_g)
    outputs_gender = Dense(1,activation='sigmoid',name='outputs_gender')(dp1_g)

    fc1_a = Dense(512)(flatten)
    fc1_a = SReLU()(fc1_a)
    dp1_a = Dropout(0.5)(fc1_a)
    outputs_age = Dense(1,activation='linear',name='outputs_age')(dp1_a)

    outputs = Dense(12,activation='softmax',name='outputs')(dp2)
    
    inputs = [
                inputs,
                inputs_p,
                inputs_d,
            ]

    outputs = [
                outputs,
                outputs_gender,
                outputs_age,
            ]

    model = Model(input=inputs, output=outputs)
    nadam = Nadam(lr=1e-4)
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(
    #             optimizer=sgd,
    #             loss = 'categorical_crossentropy',
    #             )
    
    model.compile(
                optimizer=nadam,
                loss={'outputs': 'categorical_crossentropy', 'outputs_gender': 'binary_crossentropy','outputs_age':'mse'},
                loss_weights={'outputs': 1., 'outputs_gender': 1,'outputs_age':1}
              )
    
    model_name = 'cnn_%s.hdf5'%dim
    model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_outputs_loss', save_best_only=True)
    plot(model, to_file=path+'%s.png'%model_name.replace('.hdf5',''),show_shapes=True)
    
    nb_epoch = 20
    batch_size = 128
    load_model = False
    
    if load_model:
        print('Load Model')
        model.load_weights(path+model_name)
    
    events = pd.read_pickle(path+'events_transform.pkl')
    tr_gen = X_train_generatetor_infinite(dim=dim,maxlen=maxlen,batch_size=batch_size,name="X_train.csv",events=events)
    te_gen = X_train_generatetor_infinite(dim=dim,maxlen=maxlen,batch_size=batch_size,name="X_test.csv",events=events)
    
    model.fit_generator(
        tr_gen, 
        samples_per_epoch=X_train.shape[0], 
        nb_epoch=nb_epoch, 
        verbose=1, 
        callbacks=[model_checkpoint], 
        validation_data=te_gen, 
        nb_val_samples=X_test.shape[0], 
        class_weight={}, 
        max_q_size=10
        )
    
    X_train = pd.read_csv(path+'gender_age_train.csv')
    group_le = LabelEncoder()
    group_lb = LabelBinarizer()
    labels = group_le.fit_transform(X_train['group'].values)
    labels = group_lb.fit_transform(labels)


    device_id = pd.read_csv(path+'gender_age_test.csv')['device_id']

    test = get_test(dim=dim,maxlen=maxlen,name='gender_age_test.csv',events=events)
    y_preds = model.predict(test)[0]

    # Write results
    submission = pd.DataFrame(y_preds, columns=group_le.classes_)
    submission["device_id"] = device_id
    submission = submission.set_index("device_id")
    submission.to_csv('submission_%s.csv'%dim, index=True, index_label='device_id')

if __name__ == '__main__':
    get_train_valid()
    events = get_events()
    pd.to_pickle(events,path+'events_transform.pkl')
    main()

    
