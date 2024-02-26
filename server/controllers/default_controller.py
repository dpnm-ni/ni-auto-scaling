import connexion
import six
import time
from server import util
from auto_scaling import *
from server.models.scaling_info import DQN_ScalingInfo
from server.models.scaling_info import Threshold_ScalingInfo
from server.models.scaling_info import AutoScaler

def measure_response_time():

    for i in range(0, 20):
        test_measure_response_time()

    return "sucess"

def build_test_environment():

    response = setup_env_for_test()

    return response

def get_all_scaling():

    response = []

    for process in scaler_list:
        response.append(process.get_info())

    print("get_all_scaling : ", response)

    return response

def get_scaling(name):
    response = [ process.get_info() for process in scaler_list if process.scaling_name == name]

    return response

def create_threshold_scaling(body):
    if connexion.request.is_json:
        body = Threshold_ScalingInfo.from_dict(connexion.request.get_json())
        response = AutoScaler(body, "threshold")
        scaler_list.append(response)

        threading.Thread(target=threshold_scaling, args=(response,)).start()

    return response.get_info()

def create_dqn_scaling(body):
    index = -1
    if connexion.request.is_json:
        body = DQN_ScalingInfo.from_dict(connexion.request.get_json())

        for process in scaler_list:
            if process.sfc_name == body.sfc_name:
                print("duplicated sfc_name occured")
                return

        response = AutoScaler(body, "dqn")
        scaler_list.append(response)

        threading.Thread(target=dqn_scaling, args=(response,)).start()

    return response.get_info()

def delete_scaling(name):
    index = -1
    response = []

    for process in scaler_list:
        if process.scaling_name == name:
            index = scaler_list.index(process)
            break

    if index > -1:
        response = scaler_list[index].get_info()
        scaler_list[index].set_active_flag(False)
        delete_monitor(scaler_list[index])
        scaler_list.remove(scaler_list[index])

    return response
