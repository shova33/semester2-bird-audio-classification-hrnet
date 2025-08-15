import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger, ReduceLROnPlateau
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from scipy import interp

from HR_Net import seg_hrnet
from data_gen_mc import DataReader

def plot_history(history, result_dir):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.show()
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.show()
    plt.close()
    
           
def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'History_result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

#https://www.kaggle.com/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras
def normalize(img):
    '''
    Normalizes an array 
    (subtract mean and divide by standard deviation)
    '''
    eps = 0.001
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
    return img       

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    nb_batch = 32
    num_epoch =1000
    
    '''
    #Size of the inputs
    img_rows, img_cols, frames, vid_channel  = 128, 128, 64, 3
    
    #Size of the inputs
    #video_input_fast = (img_rows, img_cols, frames_fast, vid_channel)
    video_input = (img_rows, img_cols, frames, vid_channel)
    #face_input_fast = (img_rows, img_cols, frames_fast, vid_channel)
    '''
    audio_input = (64, 503, 1)
    
    #Define the output path
    #output = 'C:/Users/YASH/Desktop/FINETUNE_BEST/VideoSF_MMTM_Music2D/'  
    output = 'D:/MV_MC_OUTCOME/RELU2_5L_SGD_Audio/'  
    create_folder(output)
    
    ######################## Loading data ###############################
    data_reader = DataReader()
    
    train_generator = data_reader.generator_train(nb_batch)
    nb_train_samples = data_reader.train_files_count()
    train_steps_per_epoch= int(nb_train_samples // nb_batch)
    
    val_generator = data_reader.generator_val(nb_batch)
    nb_val_samples = data_reader.val_files_count()
    val_steps_per_epoch= int(nb_val_samples // nb_batch)
    
    ######################## Network training Start ###############################
    
    #filepath=output +"AE_AV_model_BOOST_UP2.hdf5"
    filepath=output +"VideoMC_Music2D_BEST.h5"
    csv_logger = CSVLogger(os.path.join(output + "CSV_Logger" + '-' + 'MC_ML' + str(time.time()) + '.csv'))
    Earl_Stop = EarlyStopping(patience=15)
    #tensorboard = TensorBoard(log_dir="Audio_model_Summary/{}".format(time.time()))
    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    checkpoint = ModelCheckpoint(filepath , monitor='val_accuracy', verbose=1,save_best_only=True,save_weights_only=False, mode='max',period=1)
    #callbacks_list = [checkpoint,tensorboard,reduce_lr, csv_logger, Earl_Stop]
    callbacks_list = [checkpoint, reduce_lr, Earl_Stop, csv_logger]
    
    #ImgHeight, ImgWidth, NumChannels, NumClass = 128, 1292, 1, 8
    #network = seg_hrnet(nb_batch, ImgHeight, ImgWidth,  NumChannels, NumClass)
    network = rhythm_net(audio_input)
    #network = Audio_net(audio_input)
    history = network.fit_generator(generator=train_generator, steps_per_epoch=train_steps_per_epoch, epochs=num_epoch,
                    validation_data=val_generator, validation_steps=val_steps_per_epoch, callbacks=callbacks_list)
      
    model_json = network.to_json()
    
    if not os.path.isdir(output):
        os.makedirs(output)
    with open(os.path.join(output, 'VideoMMTM_Music2D_model.json'), 'w') as json_file:
        json_file.write(model_json)
    
    network.save_weights(os.path.join(output, 'VideoMMTM_Music2D_model.h5'))

    plot_history(history, output)
    save_history(history, output)
    
    # Data testing began    
    logmel_spc, video3D, target_h1, target_h2 = data_reader.generator_test(nb_batch)
    #video3D_C3D = normalize(video3D_C3D)
    logmel_spc = normalize(logmel_spc)
    video3D = normalize(video3D)
    
    loss, acc = network.evaluate(x=logmel_spc, y=target_h1, batch_size=5, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    
    predictions = network.predict(x=[logmel_spc, video3D], batch_size=10, verbose=0)    #x=X_test
    
    def get_confusion_matrix_one_hot(model_results, truth):
        assert model_results.shape == truth.shape
        num_outputs = truth.shape[1]
        confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
        predictions = np.argmax(model_results, axis=1)
        assert len(predictions) == truth.shape[0]
        
        for actual_class in range(num_outputs):
            idx_examples_this_class = truth[:, actual_class] == 1
            prediction_for_this_class = predictions[idx_examples_this_class]
            for predicted_class in range(num_outputs):
                count = np.sum(prediction_for_this_class == predicted_class)
                confusion_matrix[actual_class, predicted_class] = count            
        assert np.sum(confusion_matrix) == len(truth)
        assert np.sum(confusion_matrix) == np.sum(truth)
        
        return confusion_matrix
    
    
    confusion_matrix = get_confusion_matrix_one_hot(predictions, target_h1) #y_test
    print(confusion_matrix)
    
    #If you want to save prediction
    with open(output + 'ROC.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    
    
    def plot_ROC():
        SM_pred_probs = predictions
        n_samples = np.min([len(SM_pred_probs)])
        
        def plot_roc_curves(y_true, pred_probs):
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            NUM_CLASSES = 8
            for i in range(8):
                fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])    
        
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
        
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(NUM_CLASSES):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
            # Finally average it and compute AUC
            mean_tpr /= NUM_CLASSES
        
            return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)
        
        # Plot all ROC curves
        plt.figure(figsize=(6, 6), dpi=300)
        #plt.figure(figsize=(10,9))
        
        plt.title('Macro-average ROC curves')
        
        fpr, tpr, roc_auc = plot_roc_curves(target_h1[:n_samples], SM_pred_probs[:n_samples])
        plt.plot(fpr, tpr,label='Softmax Classifier (Area = {0:0.3f})'''.format(roc_auc), color='red', linestyle=':', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        #plt.xlabel('False Positive Rate', fontsize=9.5, family ='Times New Roman') # Change fornt in specific line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of Music Video Emotion')  #Receiver Operating Characteristic (ROC) Curve
        plt.legend(loc="lower right")
        plt.savefig(output + 'roc-curve.png')
        plt.savefig(output + 'roc-curve.pdf')
        
        #plt.show()
    
    print('Classifier result save in disk ')
    plot_ROC()
    print('Ploted ROC for multi-class')
    
    print('Softmax Classifier ROC AUC score= {0:.3f}'.format(roc_auc_score(y_true=target_h1, y_score=predictions, average='macro')))
    
    
    ####################################PLOT CONFUSION MATRIX####################################################################
    from sklearn import metrics
    import seaborn as sns
    
    LABELS =['African','Arabic','Chinese','French','Indian','Mongolian',
                   'Nepali','Spanish']
    
    def show_confusion_matrix(validations, predictions):
        matrix = metrics.confusion_matrix(validations, predictions)
        plt.figure(figsize=(7, 6), dpi = 300)
        sns.heatmap(matrix,
                    cmap="coolwarm",
                    linecolor='white',
                    linewidths=1,
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    annot=True,
                    fmt="d")
        plt.title("Confusion Matrix", fontsize=15)
        plt.ylabel("True Label", fontsize=10)
        plt.xlabel("Predicted Label", fontsize=10)
        plt.savefig(output + 'Confusion_matrix.png')
        plt.savefig(output + 'Confusion_matrix.pdf')
        #plt.show()
    
    print("\n--- Confusion matrix for test data ---\n")
    
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(predictions, axis=1)
    max_y_test = np.argmax(target_h1, axis=1)
    
    show_confusion_matrix(max_y_test, max_y_pred_test)
    