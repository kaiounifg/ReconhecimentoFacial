import os

# Função auxiliar para pegar o caminho da pasta de modelos
def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), "models", filename)

def pose_predictor_model_location():
    return get_model_path("shape_predictor_68_face_landmarks.dat")

def pose_predictor_five_point_model_location():
    return get_model_path("shape_predictor_5_face_landmarks.dat")

def face_recognition_model_location():
    return get_model_path("dlib_face_recognition_resnet_model_v1.dat")

def cnn_face_detector_model_location():
    return get_model_path("mmod_human_face_detector.dat")