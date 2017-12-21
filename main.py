import csv
import glob
import librosa
import math
import numpy as np
import os
import sys


def select_music_in_directory(from_dir):

    ext_list = ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']
    files = []
    for ext_itr in ext_list:
        tmp_files = glob.glob(from_dir + '/*' + ext_itr)
        if tmp_files != np.array([]):
            for file_itr in enumerate(tmp_files):
                files.append(file_itr)

    return files


def make_directory(to_dir):
    for i in range(200):

        file_index = ""
        index_val = 1

        if i + index_val < 10:
            file_index = "0000" + str(i + index_val)
        elif i + index_val < 100:
            file_index = "000" + str(i + index_val)
        elif i + index_val < 1000:
            file_index = "00" + str(i + index_val)
        elif i + index_val < 10000:
            file_index = "0" + str(i + index_val)
        else:
            file_index = str(i + index_val)

        path = to_dir + "cut_by_IMV133_video" + file_index
        os.mkdir(path)


def extract_acoustic_features():

    frame_length = 2048
    hop_length = 512

    from_dirs = ["C:/MUSIC_RECOMMENDATION/src_data/OMV62of65'audio/",
                 "C:/MUSIC_RECOMMENDATION/src_data/OMV200'audio/"]

    to_dirs = ["C:/MUSIC_RECOMMENDATION/src_data/train_features/OMV62of65_npy_frame_40aco/",
               "C:/MUSIC_RECOMMENDATION/src_data/train_features/OMV200_npy_frame_40aco/"]

    for from_dir, to_dir in zip(from_dirs, to_dirs):
        for index, file in enumerate(os.listdir(from_dir)):

            # audio_file = 'C:/Users/LAB/Music/02ParanoidAndroid.mp3'
            file_path = from_dir + file
            src_signal, sr = librosa.load(file_path, sr=44100)

            # chroma
            chroma = librosa.feature.chroma_cqt(y=src_signal, sr=sr, hop_length=hop_length)

            # mfcc
            mfcc20 = librosa.feature.mfcc(src_signal, sr=sr, n_mfcc=20)

            # spectra_contrast
            spec_contrast8 = librosa.feature.spectral_contrast(y=src_signal, sr=sr, n_fft=frame_length, hop_length=hop_length,
                                                               n_bands=7, fmin=200, quantile=0.04, linear=True)

            frame_features = np.concatenate([chroma, mfcc20, spec_contrast8], axis=0)
            frame_features = np.transpose(frame_features)

            tmp_npy_name = file.split(".")
            npy_name = tmp_npy_name[0] + "_frame_40aco.npy"
            np.save(to_dir+npy_name, frame_features)
            print("shape = ", frame_features.shape)
            print(index+1, "曲目が終了しました")

            # # cepstrum
            # spectorogram = np.abs(librosa.stft(src_signal, n_fft=frame_length, hop_length=hop_length))**2
            # tmp_cepstrum = librosa.power_to_db(spectorogram)
            # cepstrum = librosa.istft(tmp_cepstrum, hop_length=hop_length)

            # if len(cepstrum) != spectorogram.size:
            #     diff = spectorogram.size - len(cepstrum)
            #     zero_arr = np.zeros(diff)
            #     cepstrum = np.concatenate([cepstrum, zero_arr], axis=0)
            #
            # cepstrum = cepstrum.reshape(spectorogram.shape[0], spectorogram.shape[1])

        break



if __name__ == '__main__':


    # add_histogram_and_convert_frame2shot_features_for_training(threshold=10)
    extract_acoustic_features()
    # make_directory("C:/MUSIC_RECOMMENDATION/src_data/shots_train_threshold_themes/")
    # for_training_convert_frame2shot_features(threshold=10)

    # csv2npy(from_dir='C:/MUSIC_RECOMMENDATION/src_data/train_features/csv_frame_46aco_12chroma_20mfcc_14spectcontrast/',
    #         to_dir='C:/MUSIC_RECOMMENDATION/src_data/train_features/npy_frame_46aco/')
    #
    # csv2npy(from_dir='C:/MUSIC_RECOMMENDATION/src_data/train_features/csv_frame_40aco/',
    #         to_dir='C:/MUSIC_RECOMMENDATION/src_data/train_features/npy_frame_40aco/')
    # make_directory("C:/MUSIC_RECOMMENDATION/src_data/recommendation_test_features/for_test_IMV133_npy_shot_80aco/")














