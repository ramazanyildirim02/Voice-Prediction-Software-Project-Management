import joblib
from ml_model import predict 
from acc_f1score import scores
import pyaudio 
import wave
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
import librosa
import librosa.display

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def recording_audio():
    """5 saniyelik bir ses kaydı alır.

    Alınan ses kaydı proje temel dizinine kaydedilir.
    Ses dosyası .vaw uzantılı olacak şekilde yazılır.
    """

    pa = pyaudio.PyAudio()

    # Ses kaydı için ses akışını açın
    stream = pa.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=FRAMES_PER_BUFFER)

    print("Kayıt başladı...")
    seconds = 6
    frames = []
    second_tracking = 0
    second_count = 0

    for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
        second_tracking += 1
        if second_tracking == RATE/FRAMES_PER_BUFFER:
            second_count += 1
            second_tracking = 0
            print(f"time left: {seconds - second_count}")



    stream.stop_stream()
    stream.close()
    pa.terminate()


    obj = wave.open("output_audio.wav", "wb")
    obj.setnchannels(CHANNELS)
    obj.setsampwidth(pa.get_sample_size(FORMAT))
    obj.setframerate(RATE)
    obj.writeframes(b"".join(frames))
    obj.close()

    file = wave.open("output_audio.wav", "rb")


    sample_freq = file.getframerate()
    frames = file.getnframes()
    signal_wave = file.readframes(-1)

    file.close()



def speech_to_text(audio_file):
    """Ses dosyasını metne çevirir.

    Google Text API kullanarak ses dosyasını Türkçe metne çevirir.

    Args:
        audio_file (Wave): Ses dosyası
    """

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)  # Ses dosyasını kaydedin

        try:
            text = recognizer.recognize_google(audio_data, language="tr-TR")
            print("Ses metni:")
            print(text)
        except sr.UnknownValueError:
            print("Ses algılanamadı!")
        except sr.RequestError as e:
            print("Google Speech Recognition API'ye erişilemiyor; {0}".format(e))
        except Exception as e:
            print("Bir hata oluştu: {0}".format(e))


def extract_features(audio_file):
    """Ses dosyalarını dataframe içerisinde nümerik değerler olarak yazdırır.

    Args:
        audio_file (Wave): Ses dosyası

    Returns:
        nparray: NumPY Array
    """

    # Ses dosyasını yükle
    y, sr = librosa.load(audio_file)
        
    # Mel-frekans spektrogramunu hesapla
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Vektörleştirme (örneğin, sütunlardaki ortalamayı alabilirsiniz)
    feature_vector = np.mean(mel_spectrogram, axis=1)

    return feature_vector


def predict_audio():
    """Sesin kime ait olduğunu tahmin eder.

    Returns:
        str: Sesin kime ait olduğu
    """

    audio_data_df = pd.DataFrame([extract_features("output_audio.wav")],columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76', 'feature_77', 'feature_78', 'feature_79', 'feature_80', 'feature_81', 'feature_82', 'feature_83', 'feature_84', 'feature_85', 'feature_86', 'feature_87', 'feature_88', 'feature_89', 'feature_90', 'feature_91', 'feature_92', 'feature_93', 'feature_94', 'feature_95', 'feature_96', 'feature_97', 'feature_98', 'feature_99', 'feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104', 'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109', 'feature_110', 'feature_111', 'feature_112', 'feature_113', 'feature_114', 'feature_115', 'feature_116', 'feature_117', 'feature_118', 'feature_119', 'feature_120', 'feature_121', 'feature_122', 'feature_123', 'feature_124', 'feature_125', 'feature_126', 'feature_127', 'feature_128'])
    
    prediction = predict(audio_data_df)
    return prediction


def load_hist(x, path, isim):
    """Ses dosyasının histogramını çıkartır.

    Args:
        x (Wave): Ses dosyası
        path (Wave): Ses dosyası
        isim (str): Sesin isim olarak kime ait olduğu
    """

    x, sr = librosa.load(path)
    
    plt.figure(figsize=(15, 17))
    plt.subplot(3, 1, 1)
    plt.plot(x)
    plt.title(f"{isim} ait tahmin")
    plt.ylim((-1, 1))
    plt.savefig('grafik.png')  # Grafiği önce kaydedin
    plt.show()  # Grafiği sonra görüntüleyin


scores()
recording_audio()
speech_to_text("output_audio.wav")
print(predict_audio())
load_hist("output_audio.wav", "output_audio.wav", predict_audio().split()[1])
