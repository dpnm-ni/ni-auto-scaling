import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from datetime import datetime, timedelta, timezone
from create_dashboard import create_dashboard
from config import cfg
from torch_dqn import *

import numpy as np
import threading
import datetime as dt
import math
import os
import time
import subprocess
from pprint import pprint
import random
import re


# Parameters
# OpenStack Parameters
openstack_network_id = cfg["openstack_network_id"] # Insert OpenStack Network ID to be used for creating SFC
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"

#ni_nfvo_client_api
ni_nfvo_client_cfg = ni_nfvo_client.Configuration()
ni_nfvo_client_cfg.host=cfg["ni_nfvo"]["host"]
ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))

#ni_monitoring_api
ni_mon_client_cfg = ni_mon_client.Configuration()
ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))


# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 2500            # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons = 128               # Number of neurons in each hidden layer
epsilon = 0.10                  # epsilon value of e-greedy algorithm
required_mem_size = 20          # Minimum number triggering sampling
print_interval = 20             # Number of iteration to print result during DQN

scaler_list = []
sfc_update_flag = True


def setup_env_for_test():

    response = "Cannot find test-sfcrs for test"
    deployed_sfcrs = ni_nfvo_sfcr_api.get_sfcrs()

    for sfcr in deployed_sfcrs:
        if sfcr.name.startswith("test-auto-scaling"):
            if get_sfc_by_name(sfcr.name):
                continue
            else:
                print("building environment...")
                response = build_env_for_test(sfcr)  
        else:
            continue
            
    return response


def build_env_for_test(sfcr):

    #Test environment
    target_nodes_0 = ["ni-compute-181-155","ni-compute-181-155","ni-compute-181-158","ni-compute-181-158","ni-compute-181-158"]
    target_nodes_1 = ["ni-compute-181-155","ni-compute-181-156","ni-compute-181-158","ni-compute-181-158","ni-compute-181-203"]
    target_nodes_2 = ["ni-compute-181-203","ni-compute-181-158","ni-compute-181-156","ni-compute-181-155","ni-compute-181-155"]
    target_nodes = [target_nodes_0, target_nodes_1, target_nodes_2]
    
    #Check name is 0, 1, or 2
    idx = int(re.search(r'\d+$', sfcr.name).group())
    
    target_node = target_nodes[idx]
    target_type = sfcr.nf_chain
    sfc_in_instance_id =[]
    sfc_in_instance = []     
    
    for j in range(0, len(target_type)):
        print("{} {} {}".format(target_type[j], target_node[j], sfcr.name + cfg["instance"]["prefix_splitter"]+target_type[j]))
        vnf_spec = set_vnf_spec(target_type[j], target_node[j], sfcr.name + cfg["instance"]["prefix_splitter"]+target_type[j])
        vnf_id = deploy_vnf(vnf_spec)

        limit = 500 
        for i in range(0, limit):
            time.sleep(2)

            # Success to create VNF instance
            if check_active_instance(vnf_id):
                sfc_in_instance.append([ni_mon_api.get_vnf_instance(vnf_id)])
                sfc_in_instance_id.append([vnf_id])
                break
            elif i == (limit-1):
                print("destroy vnf")
                destroy_vnf(vnf_id)
    
    ##Stress inject
    
    create_sfc(sfcr, sfc_in_instance_id)
    AS_mydashboard_url = create_dashboard(sfc_in_instance,"Auto-scaling-VNF")
    
    return ("Target sfc : {} ML grafana dashboard : {}".format(sfcr.name, AS_mydashboard_url))
    

def create_sfc(sfcr, instance_id_list):

    sfc_spec =ni_nfvo_client.SfcSpec(sfc_name=sfcr.name,
                                 sfcr_ids=[sfcr.id],
                                 vnf_instance_ids=instance_id_list,
                                 is_symmetric=False)


    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)
    print("Success to pass for creating sfc")
    return api_response

def set_vnf_spec(vnf_type, node_name, vnf_name):
    vnf_spec = get_nfvo_vnf_spec2(vnf_type)
    vnf_spec.vnf_name = vnf_name
    vnf_spec.image_id = cfg["image"][vnf_type]
    vnf_spec.node_name = node_name

    return vnf_spec 


def get_nfvo_vnf_spec2(flavor_name):

    t = ni_nfvo_client.ApiClient(ni_nfvo_client_cfg)

    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(t)
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"][flavor_name]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


# get_nfvo_vnf_spec(): get ni_nfvo_vnf spec to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf spec
def get_nfvo_vnf_spec():

    t = ni_nfvo_client.ApiClient(ni_nfvo_client_cfg)

    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(t)
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"]["default"]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


# get_ip_from_vm(vm_id): get a data plane IP of VM instance
# Input: vm instance id
# Output: port IP of the data plane
def get_ip_from_id(vm_id):
#    print("6")

    query = ni_mon_api.get_vnf_instance(vm_id)
    #print(query)

    ## Get ip address of specific network
    ports = query.ports
    #print(ports)
    network_id = openstack_network_id
    #print(network_id)

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


# get_node_info(): get all node information placed in environment
# Input: null
# Output: Node information list
def get_node_info():
    query = ni_mon_api.get_nodes()

    response = [ node_info for node_info in query if node_info.type == "compute" and node_info.status == "enabled"]
    response = [ node_info for node_info in response if not (node_info.name).startswith("NI-Compute-82-9")]

    return response



def get_vnf_info(sfc_info):
    all_vnf_info = ni_mon_api.get_vnf_instances()
    selected_vnfi = []
    
    for vnf_ids in sfc_info.vnf_instance_ids:
        for vnf_id in vnf_ids:
            selected_vnfi.append(ni_mon_api.get_vnf_instance(vnf_id))

    return selected_vnfi


# get_sfcr_by_name(sfcr_name): get sfcr information by using sfcr_name from NFVO module
# Input: sfcr name
# Output: sfcr_info
def get_sfcr_by_name(sfcr_name):
#    print("9")
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.name == sfcr_name ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info


# get_sfcr_by_id(sfcr_id): get sfc information by using sfcr_id from NFVO module
# Input: sfcr_id, FYI, sfcr is a flow classifier in OpenStack
# Output: sfcr_info
def get_sfcr_by_id(sfcr_id):

    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.id == sfcr_id ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info


# get_sfc_by_name(sfc_name): get sfc information by using sfc_name from NFVO module
# Input: sfc name
# Output: sfc_info
def get_sfc_by_name(sfc_name):
#    print("11")

    query = ni_nfvo_sfc_api.get_sfcs()

    sfc_info = [ sfci for sfci in query if sfci.sfc_name == sfc_name ]

    if len(sfc_info) == 0:
        return False

    sfc_info = sfc_info[-1]

    return sfc_info


# get_soruce_client(sfc_name): get source client ID by using sfc_name
# Input: sfc name
# Output: source client info
def get_source_client(sfc_name):
#    print("12")

    sfc_info = get_sfc_by_name(sfc_name)
    sfcr_info = get_sfcr_by_id(sfc_info.sfcr_ids[0])
    source_client = ni_mon_api.get_vnf_instance(sfcr_info.source_client)

    return source_client


# get_soruce_client(sfc_name): get source client ID by using sfc_name
# Input: sfc name
# Output: source client info
def get_destination_client(sfc_name):
#    print("12")

    sfc_info = get_sfc_by_name(sfc_name)
    sfcr_info = get_sfcr_by_id(sfc_info.sfcr_ids[0])
    destination_client = ni_mon_api.get_vnf_instance(sfcr_info.destination_client)

    return destination_client

# get_tier_status(vnf_info, sfc_info, source_node): get each tier status, each tier includes same type VNF instances
# Input: vnf_info, sfc_info, SFC's source_node
# Output: each tier status showing resource utilizataion (CPU, Memory, Number of disk operations), size, distribution)
def get_tier_status(vnf_info, sfc_info, source_node):
#    print("13")

    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge",
                     "vda___disk_ops___read___derive",
                     "vda___disk_ops___write___derive"]

    vnf_instances_ids = sfc_info.vnf_instance_ids
    tier_status = []
    num_nodes = len(get_node_info())

    # Set time-period to get resources
    end_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
    start_time = end_time - dt.timedelta(seconds = 10)

    # Select each tier (VNF type)
    for tier_vnfs in vnf_instances_ids:
        tier_values = []
        tier_distribution = []

        # Select each instance ID in each tier
        for vnf in tier_vnfs:
            vnf_values = []

            for vnfi in vnf_info:
                if vnfi.id == vnf:
                    tier_distribution.append(vnfi.node_id)

            # Calculate resource status of each vnf
            for type in resource_type:

                query = ni_mon_api.get_measurement(vnf, type, start_time, end_time)
                value = 0
                memory_total = 0

                for response in query:
                    value = value + response.measurement_value

                num_query = len(query)
                value = 0 if num_query == 0 else value/num_query

                # Memory utilization calculation
                if type.startswith("memory"):
                    flavor_id = ni_mon_api.get_vnf_instance(vnf).flavor_id
                    memory_ram_mb = ni_mon_api.get_vnf_flavor(flavor_id).ram_mb
                    memory_total = 1000000 * memory_ram_mb

                    value = 100*(1-(value/memory_total)) if num_query != 0 else 0

                vnf_values.append(value) # CPU, Memory, Disk utilization of each VNF

            tier_values.append(vnf_values) # CPU, Memory, Disk utilization of each tier

        # Define status
        cpu, memory, disk = 0, 0, 0

        # Calculate sum of each resource utilization of each tier
        for vnf_values in tier_values:
            cpu = cpu + vnf_values[0]
            memory = memory + vnf_values[1]
            disk = disk + vnf_values[2] + vnf_values[3]

        # Calculate average resource utilization to define state
        tier_size = len(tier_values)

        resource = 0.70*cpu/tier_size + 0.30* memory/tier_size #+ 0.001*disk/tier_size
        tier_nodes = tier_distribution

        dist_tier = 0

        for node in tier_nodes:
            num_hops = get_hops_in_topology(source_node, node)
            dist_tier = dist_tier + num_hops/tier_size

        status = { "resource" : resource, "size" : tier_size, "distribution" : dist_tier, "placement": tier_nodes }

        tier_status.append(status)

    return tier_status


# get_state(tier_status): pre-processing tier_status to make tensor as an input data
# Input: tier_status
# Output: np.array (tensor for input values)
def get_state(tier_status):
#        print("14")
        s = []
        for tier in tier_status:
            s.append(tier["resource"])
            s.append(tier["size"])
            s.append(tier["distribution"])

        return np.array(s)


# get_target_tier(tier_status, flag): get target tier to be applied auto-scaling action
# Input: tier_status, flag (flag is a value to check scaling in or out, positive number is scaling out)
# Output: tier index (if index is a negative number, no tier for scaling)
def get_target_tier(tier_status, flag, random_flag):
#    print("15")

    tier_scores = []

    for tier in tier_status:
        resource_utilization = tier["resource"]
        size = tier["size"]
        dist_tier = tier["distribution"]

        if flag > 0: # Add action # Scale-out
            scaling_mask = 0.0 if size > 4 else 1.0
            dist_tier = 1.0 if dist_tier == 0 else dist_tier
            score = math.exp(resource_utilization)
            score = scaling_mask*(1/dist_tier)*score
        else: # Remove action # Scale-in
            scaling_mask = 0.0 if size < 2 else 1.0
            dist_tier = 1.0 if dist_tier == 0 else dist_tier
            score = math.exp(-resource_utilization)
            score = scaling_mask*dist_tier*score

        tier_scores.append(score)

    # Target tier has the highest value
    high_score = max(tier_scores)

    if high_score == 0: # No tier for scaling
        return -1

    if not random_flag:
        # Target selection
        return tier_scores.index(max(tier_scores))

    else:
        # Random selection
        list_for_random = [score for score in tier_scores if score > 0 ]
        return tier_scores.index(random.choice(list_for_random))


# get_scaling_target(tier_status, flag): get target to to be applied auto-scaling action
# Input: tier_status, source_node, flag (flag is a value to check scaling in or out, positive number is scaling out)
# Output: target (node id in case of scale-out, vnf id in case of scale-in)
def get_scaling_target(tier_status, source_node, flag, random_flag):
#    print("16")

    node_info = get_node_info()
    tier_nodes = tier_status["placement"]
    target_dist = []

    if flag > 0: # scaling-out
        for node in node_info:
            if check_available_resource(node.id):
                hop = get_hops_in_topology(source_node, node.id)
                dist_tier = ((tier_status["distribution"] * tier_status["size"]) + hop) / (tier_status["size"] + 1)
                target_dist.append(dist_tier)

            else:
                target_dist.append(10000)

        if not random:
            # Target node has the highest value
            return node_info[target_dist.index(min(target_dist))].id
        else:
            # Random selection
            return random.choice(node_info).id

    else:
        for node in tier_nodes:
            hop = get_hops_in_topology(source_node, node)
            dist_tier = ((tier_status["distribution"] * tier_status["size"]) - hop) / (tier_status["size"] - 1)
            target_dist.append(dist_tier)

        if not random_flag:
            # Target instance has the highest value
            return target_dist.index(min(target_dist))

        else:
            # Random selection
            return random.randrange(0,len(target_dist))



def deploy_vnf(vnf_spec):

    api_response = ni_nfvo_vnf_api.deploy_vnf(vnf_spec)
    print("check deployed")
    print(vnf_spec)
    print(api_response)

    return api_response



def destroy_vnf(id):

    api_response = ni_nfvo_vnf_api.destroy_vnf(id)

    return api_response


# measure_response_time(): send http requests from a source to a destination
# Input: scaler
# Output: response time
def measure_response_time(scaler, name):
#    print("19")

    cnd_path = os.path.dirname(os.path.realpath(__file__))

    dst_ip = get_ip_from_id(scaler.get_monitor_dst_id())
    src_ip = get_ip_from_id(scaler.get_monitor_src_id())
    #print(dst_ip)


    command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_http_e2e.sh %s %s %s %s %s" % (cfg["traffic_controller"]["password"],
                                                                            cfg["traffic_controller"]["username"],
                                                                            cfg["traffic_controller"]["ip"],
                                                                            src_ip,
                                                                            cfg["instance"]["username"],
                                                                            cfg["instance"]["password"],
                                                                            cfg["traffic_controller"]["num_requests"],
                                                                            dst_ip))


    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"


    print(command)

    # Wait until web server is running
    start_time = dt.datetime.now()
    print(start_time)

    while True:
        #print("19-while loop")
        time.sleep(10)
        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        print(response)
        if response != "":
            #print("if")
            pprint("[%s] %s" % (scaler.get_scaling_name(), response))
            f = open("test_monitor-"+name+".txt", "a+", encoding='utf-8')
            f.write(str(response)+'\n')
            f.close()
            print("write done")
            return float(response)
        elif (dt.datetime.now() - start_time).seconds > 60 or scaler.get_active_flag() == False:
            #print("elif")
            scaler.set_active_flag(False)
            return -1


# lable_resource(flavor_id): check whether there are enough resource in nodes
# Input: node_id
# Output: True (enough), False (otherwise)
def check_available_resource(node_id):
#    print("20")

    node_info = get_node_info()
    selected_node = [ node for node in node_info if node.id == node_id ][-1]
    flavor = ni_mon_api.get_vnf_flavor(cfg["flavor"]["default"])

    if selected_node.n_cores_free >= flavor.n_cores and selected_node.ram_free_mb >= flavor.ram_mb:
        return True

    return False


# create_monitor(scaler): create instances to be source and destination to measure response time
# Input: scaler
# Output: True (success to create instances), False (otherwise)
def create_monitor(scaler):
#    print("21")
    source_client = get_source_client(scaler.get_sfc_name())
    destination_client = get_destination_client(scaler.get_sfc_name())

    #print("source client.id : ", source_client.id)
    scaler.set_monitor_src_id(source_client.id)
    scaler.set_monitor_dst_id(destination_client.id)

    #print("sfcr.id : ", get_sfc_by_name(scaler.get_sfc_name()).sfcr_ids[-1])
    scaler.set_monitor_sfcr_id(get_sfc_by_name(scaler.get_sfc_name()).sfcr_ids[-1])

    return True


# delete_monitor(scaler): delete SLA monitor instances
# Input: scalerdelete_monitor
# Output: Null
def delete_monitor(scaler):
#    print("22")

    scaler.set_monitor_src_id("")
    scaler.set_monitor_dst_id("")
    scaler.set_monitor_sfcr_id("")   


# update_sfc(sfc_info): Update SFC, main function to do auto-scaling
# Input: updated sfc_info, which includes additional instances or removed instances
# Output: Boolean
def update_sfc(sfc_info):
    #print("25")
    sfc_update_spec = ni_nfvo_client.SfcUpdateSpec() # SfcUpdateSpec | Sfc Update info.

    sfc_update_spec.sfcr_ids = []#only decalre when sfc_info.sfcr_ids is changed
    sfc_update_spec.vnf_instance_ids = sfc_info.vnf_instance_ids

    for i in range(0, 15):
        time.sleep(4)

        global sfc_update_flag

        if sfc_update_flag:
            sfc_update_flag = False
            print("-------sfc_info/sfc_update_spec------------")
            print(sfc_info)
            #print(sfc_info.id)
            print(sfc_update_spec)
            print("-------------------------------------------")
            ni_nfvo_sfc_api.update_sfc(sfc_info.id, sfc_update_spec)
            sfc_update_flag = True

            return True

    return False


# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
    status = ni_mon_api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


# calculate_reward(vnf_info, sfc_info, tier_status, slo, response_time): calcuate reward about action
# Input: vnf_info, node_info, sfc_info, tier_status, slo, response_time (get data to calculate reward)
# Output: calculated reward
def calculate_reward(vnf_info, sfc_info, tier_status, slo, response_time):
#    print("29")
    alpha = 1.0 # weight1
    beta =  1.2 # weight2
    sla_score = 0 # Check sla violation
    dist_sfc = 0 # Distribution of SFC

    # Preprocessing: get vnf IDs placed in sfc
    sfc_vnf_ids = []
    tier_size = len(sfc_info.vnf_instance_ids)

    for vnf_ids in sfc_info.vnf_instance_ids:
        sfc_vnf_ids = sfc_vnf_ids + vnf_ids

    # SLA violation check for reward
    # VNF Usage (Total VNF / tierSize) and Dist_SFC
    sla_score = -response_time/slo
    vnf_usage = len(sfc_vnf_ids)/tier_size

    for status in tier_status:
        dist_sfc = dist_sfc + status["distribution"]/tier_size

    # Calculation reward
    reward = sla_score + (alpha * (1/vnf_usage) * math.exp(-beta*dist_sfc))
    return reward

def get_instances_in_sfc(vnf_info, sfc_info):
    instance_list = []
    #print("get_instances_in_sfc")

    for type_ids in sfc_info.vnf_instance_ids:
        type_instances = [ instance for instance in vnf_info if instance.id in type_ids ]
        if type_instances != []:
            instance_list.append(type_instances) 

    #print("this is type_instances : ", instance_list)
    return instance_list

def get_instance_info(instance, flavor):
#    print("32")
    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge",
                     "vda___disk_ops___read___derive",
                     "vda___disk_ops___write___derive",
                     "___if_dropped___tx___derive",
                     "___if_dropped___rx___derive",
                     "___if_packets___tx___derive",
                     "___if_packets___rx___derive"]

    info = { "id": instance.id, "cpu" : 0.0, "memory": 0.0, "disk": 0.0, "packets": 0.0, "drops": 0.0, "loss": 0.0, "location": "NULL" }

    # Get port names
    for port in instance.ports:
        if port.network_id == openstack_network_id:
            network_port = port.port_name
            break

    for resource in resource_type:
        if "drop" in resource or "packets" in resource:
            resource_type[resource_type.index(resource)] = network_port + resource

    # Set time-period to get resources
    end_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
    start_time = end_time - dt.timedelta(seconds = 10)

    if str(end_time)[-1]!='Z':
         end_time = str(end_time.isoformat())+ 'Z'
    if str(start_time)[-1]!='Z':
         start_time = str(start_time.isoformat()) + 'Z'

    for resource in resource_type:

        
        query = ni_mon_api.get_measurement(instance.id, resource, start_time, end_time)
        value = 0

        for response in query:
            value = value + response.measurement_value

        value = value/len(query) if len(query) > 0 else 0

        if resource.startswith("cpu"):
            info["cpu"] = value
        elif resource.startswith("memory"):
            memory_ram_mb = flavor.ram_mb
            memory_total = 1000000 * memory_ram_mb
            info["memory"] = 100*(1-(value/memory_total)) if len(query) > 0 else 0
        elif resource.startswith("vda"):
            info["disk"] = info["disk"] + (value/(2*1000000))
        elif "___if_dropped" in resource:
            info["drops"] = info["drops"] + value
        elif "___if_packets" in resource:
            info["packets"] = info["packets"] + value

    info["location"] = instance.node_id
    info["loss"] = info["drops"]/info["packets"] if info["packets"] > 0 else 0

    #print("info : ", info)

    return info

def get_type_status(type_instances, flavors):
    type_status = []

    # Set time-period to get resources
    end_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
    start_time = end_time - dt.timedelta(seconds = 10)
    
    flavor_bug = []
    flavor = {}
    for flavor_info in flavors:
        flavor_bug.append(flavor_info)

     
    

    for _type in type_instances: ##Type of VNFS, so even 6vnf, 5types..
        type_info = { "cpu" : 0.0, "memory": 0.0, "disk": 0.0, "packets": 0.0, "drops": 0.0, "loss": 0.0, "location": [], "size": 0, "allocation": {"core": 0, "memory": 0} }
        type_size = len(_type)
        type_info["size"] = type_size

        for instance in _type:
            for flavor_info in flavor_bug:
                if flavor_info.id == instance.flavor_id:
                    flavor = flavor_info
                    type_info["allocation"]["core"] = flavor_info.n_cores
                    type_info["allocation"]["memory"] = flavor_info.ram_mb

                    break


            instance_info = get_instance_info(instance, flavor)
            type_info["cpu"] = type_info["cpu"] + instance_info["cpu"]/type_size
            type_info["memory"] = type_info["memory"] + instance_info["memory"]/type_size
            type_info["disk"] = type_info["disk"] + instance_info["disk"]/type_size
            type_info["packets"] = type_info["packets"] + instance_info["packets"]/type_size
            type_info["drops"] = type_info["drops"] + instance_info["drops"]/type_size
            type_info["location"].append(instance_info["location"])

            #print("instance loop fin")

        type_info["loss"] = type_info["drops"]/type_info["packets"] if type_info["packets"] > 0 else 0

        type_status.append(type_info)
    #print("type_status")
    #print(type_status)

    return type_status

def get_hops_in_topology(src_node, dst_node):
#    print("34")

    nodes = [ "ni-compute-181-155", "ni-compute-181-156", "ni-compute-181-157", "ni-compute-181-158", "ni-compute-181-203", "ni-compute-181-162", "ni-compute-kisti", "ni-compute-181-154"]
    hops = [[1, 2, 4, 4, 4, 6, 8, 10],
            [2, 1, 4, 4, 4, 6, 8, 10],
            [4, 4, 1, 2, 2, 6, 8, 10],
            [4, 4, 2, 1, 2, 6, 8, 10],
            [4, 4, 2, 2, 1, 6, 8, 10],
            [6, 6, 6, 6, 6, 1, 8, 10],
            [8, 8, 8, 8, 8, 8, 1, 10],
            [10, 10, 10, 10, 10, 10, 10, 1]]


    return hops[nodes.index(src_node)][nodes.index(dst_node)]


def state_pre_processor(service_info):
#    print("35")

    state = []
    #print(service_info)

    # Create state
    state.append(service_info["cpu"])
    state.append(service_info["memory"])
    state.append(service_info["disk"])
    state.append(service_info["drops"]/service_info["packets"] if service_info["packets"] > 0 else 0)
    state.append(service_info["placement"])

    return np.array(state)

def get_service_info(vnf_info, sfc_info, flavors):
    
    type_instances = get_instances_in_sfc(vnf_info, sfc_info)
    type_status = get_type_status(type_instances, flavors)
    
   
    #print("type_status : ", type_status)
    
    
    
    service_info = { "cpu" : 0.0, "memory": 0.0, "disk": 0.0, "packets": 0.0, "drops": 0.0, "location": [], "placement": 0.0, "num_types": 0, "size": 0 }
    size = len(type_status)

    #print("size : ", size)

    for status in type_status:
        service_info["cpu"] = service_info["cpu"] + status["cpu"]/size
        service_info["memory"] = service_info["memory"] + status["memory"]/size
        service_info["disk"] = service_info["disk"] + status["disk"]/size
        service_info["packets"] = service_info["packets"] + status["packets"]/size
        service_info["drops"] = service_info["drops"] + status["drops"]/size
        service_info["location"] = service_info["location"] + status["location"]
        service_info["num_types"] = service_info["num_types"] + 1

    # Placement
    placement_value = 0
    source_place = get_source_client(sfc_info.sfc_name).node_id
    size = len(service_info["location"])

    for location in service_info["location"]:
        placement_value = placement_value + (get_hops_in_topology(source_place, location)/size)

    service_info["size"] = size
    service_info["placement"] = placement_value

    #print("service_info : ", service_info)

    return service_info

def get_target_type(type_status, source_client, flag, random_flag): # decision model #should add 1 for src..
#    print("37")

    type_scores = []
    alpha = 0.85
    beta = 0.15

    #print(type_status)
    for type in type_status:
        resource_utilization = (alpha*(type["cpu"]/type["allocation"]["core"])) + (beta*(type["memory"]/(type["allocation"]["memory"])))
        dist = 0

        for location in  type["location"]:
            dist = dist + (get_hops_in_topology(source_client, location)/type["size"])

        if flag > 0: # Add action
            scaling_mask = 0.0 if type["size"] > 4 else 1.0
            dist = 1.0 if dist == 0 else dist
            score = (1/dist)*math.exp(resource_utilization)
            score = scaling_mask*score
        else: # Remove action
            scaling_mask = 0.0 if type["size"] < 2 else 1.0
            dist = 1.0 if dist == 0 else dist
            score = dist*math.exp(-resource_utilization)
            score = scaling_mask*score

        type_scores.append(score)

    # Target tier has the highest value
    high_score = max(type_scores)

    if high_score == 0: # No tier for scaling
        return -1

    if not random_flag:
        # Target selection
        return type_scores.index(max(type_scores)) 

    else:
        # Random selection
        list_for_random = [score for score in type_scores if score > 0 ]
        return type_scores.index(random.choice(list_for_random)) 

def reward_calculator(service_info, response_time):
#    print("38")
    alpha = 1.0 # weight1
    beta =  1.0 # weight2
    gamma = 1.5 # weight3

    response_time = response_time/1000.0
    #response_time = response_time
    loss = service_info["drops"]/service_info["packets"] if service_info["packets"] != 0 else 1
    inst_count = service_info["size"]/(service_info["num_types"]*5)

    reward = -((alpha*math.log(1+response_time)+(beta*math.log(1+loss))+(gamma*math.log(1+inst_count))))

    print("reward value 1 : {}".format(alpha*math.log(1+response_time)))
    print("reward value 2 : {}".format(beta*math.log(1+loss)))
    print("reward value 3 : {}".format(gamma*math.log(1+inst_count)))


    return reward


###Ridiculos!! below function.....
#It is not consider Edge environment
#Also, it does not consider back-routing that increasing delay..

def get_scaling_target(status, source_node, flag, random_flag): # decision model
#    print("39")

    node_info = get_node_info()
    type_nodes = status["location"]
    target_dist = []

    # Total Dist
    # Fix as use standard vnf that initaialy installed
    standard = status["location"][0]
    
    total_dist = 0
    for location in status["location"]:
        total_dist = total_dist + get_hops_in_topology(standard, location)
        
    #Maybe we can calculate optimal routing path that not duplicating paths.
    #For example, get edge info and does not allow to go back previous edge.
    #But in this case, we also should consider migration case that scale out & in... for this problem we can handle migration problem when working for orchestration. 
            

    # Scale-out
    if flag > 0: # scaling-out
        for node in node_info:
            if check_available_resource(node.id):
                dist = (total_dist + get_hops_in_topology(standard, node.id))/(status["size"]+1)
                target_dist.append(dist)
            else:
                target_dist.append(10000)

        if not random_flag:
            # Target node has the highest value
            return node_info[target_dist.index(min(target_dist))].id
        else:
            # Random selection
            return random.choice(node_info).id

    # Scale-in
    else:
        for node in type_nodes:
            dist = (total_dist - get_hops_in_topology(standard, node))/(status["size"]-1)
            target_dist.append(dist)

        if not random_flag:
            # Target instance has the highest value
            return target_dist.index(min(target_dist))

        else:
            # Random selection
            return random.randrange(0,len(target_dist))
            
            
# dqn-threshold(scaler): doing auto-scaling based on dqn
# Input: scaler
# Output: none
def dqn_scaling(scaler):

#    print("28")
    sfc_info = get_sfc_by_name(scaler.get_sfc_name())
    print("dqn scaling")
    # Target SFC exist
    if sfc_info:
        # Initial Processing
        start_time = dt.datetime.now() #+ dt.timedelta(hours = 24)
        source_client = get_source_client(scaler.get_sfc_name())
        epsilon_value = epsilon

        flavors = ni_mon_api.get_vnf_flavors()
        instance_types = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
        #del instance_types[0] # Flow classifier instance deletion

        print("initial instance_types")
        print(instance_types)
        #node_info = get_node_info()

        # Q-networks
        num_states = 5 # Number of states
        num_actions = 3 # Add, Maintain, Remove

        q = Qnet(num_states, num_actions, num_neurons)
        q_target = Qnet(num_states, num_actions, num_neurons)
        q_target.load_state_dict(q.state_dict())
        
        
        if scaler.has_dataset == True:
            print("loaded the trained model")
            q.load_state_dict(torch.load("save_model/"+scaler.get_scaling_name()))
            q_target.load_state_dict(torch.load("save_model/"+scaler.get_scaling_name()))
            
        else:
            print("learning the data")

        optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        n_epi = 0

        # If there is dataset, read it
        memory = ReplayBuffer(buffer_limit)

        # Start scaling
        scaler.set_active_flag(create_monitor(scaler))
        print("start scaling")
        
        #Epsilon_value setting
        epsilon_value = 0.5#0.99
        
        if scaler.has_dataset == True:
            epsilon_value = 0.11
            print("has dataset")
            
        print("TEST")
        #return
        
        while scaler.get_active_flag():
            
            sfc_info = get_sfc_by_name(scaler.get_sfc_name())
            vnf_info = get_vnf_info(sfc_info)
            
            #print("vnf_info : ", vnf_info)

            service_info = get_service_info(vnf_info, sfc_info, flavors)

            # Get state and select action
            s = state_pre_processor(service_info)

            decision = q.sample_action(torch.from_numpy(s).float(), epsilon_value)

            a = decision["action"]
            decision_type = "Policy" if decision["type"] else "R"

            done = False

            # Check whether it is out or in or maintain
            if a == 0:
                print("[%s] Scaling-out! by %s" % (scaler.get_scaling_name(), decision_type))
                scaling_flag = 1
            elif a == 2:
                print("[%s] Scaling-in! by %s" % (scaler.get_scaling_name(), decision_type))
                scaling_flag = -1
            else:
                print("[%s] Maintain! by %s" % (scaler.get_scaling_name(), decision_type))
                scaling_flag = 0

            #For test!!
            #scaling_flag = 0
            # Scaling in or out
            print("Epsilon value : ", epsilon_value)
            if scaling_flag != 0:
                print("action ")
                # Scaling할 Type 선택
                
                type_instances = get_instances_in_sfc(vnf_info, sfc_info) #not include server or client
                type_status = get_type_status(type_instances, flavors) #only three
                type_index = get_target_type(type_status, source_client.node_id, scaling_flag, False)
                print("*************type_instances*************")
                print(type_instances)
                print(type_status)
                print(type_index)

                if type_index > -1:
                    scaling_target = get_scaling_target(type_status[type_index], source_client.node_id, scaling_flag, False) ###initially commented in original code
                    type_name = instance_types[type_index]
                    instance_ids_in_type = [ instance.id for instance in type_instances[type_index]]
                    num_instances = len(instance_ids_in_type)
                    

                    # Scaling-out
                    if scaling_flag > 0:
                        print("scaling out")
                        # If possible to deploy new VNF instance
                        if num_instances < cfg["instance"]["max_number"]:
                            vnf_spec = get_nfvo_vnf_spec()
                            vnf_spec.vnf_name = sfc_info.sfc_name+ cfg["instance"]["prefix_splitter"] + type_name+ "scaled"
                            vnf_spec.flavor_id = cfg["flavor"][type_name]
                            vnf_spec.image_id = cfg["image"][type_name]
                            vnf_spec.node_name = scaling_target
                            instance_id = deploy_vnf(vnf_spec)

                            # Wait 1 minute until creating VNF instnace
                            limit = 100
                            for i in range(0, limit):
                                time.sleep(2)

                                # Success to create VNF instance
                                if check_active_instance(instance_id):
                                    #tier_vnf_ids.append(instance_id)
                                    print(sfc_info)
                                    print("Checked instance actived. Waiting for setup vnf program.(for sfc load balancing)")
                                    time.sleep(30)                                    

                                    #sfc_type_index = sfc_info.vnf_instance_ids.index(instance_ids_in_type)
                                    instance_ids_in_type.append(instance_id)
                                    sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                                    update_sfc(sfc_info)
                                    done = True
                                                                        
                                    instance_info_list = []
                                    for vnf_array in sfc_info.vnf_instance_ids:
                                        instance_info_list.append([])
                                        for a_vnf in vnf_array:
                                            instance_info_list[-1].append(ni_mon_api.get_vnf_instance(a_vnf))
                                    create_dashboard(instance_info_list,"Auto-scaling-VNF")
                                    
                                    break
                                elif i == (limit-1):
                                    print("destroy vnf")
                                    destroy_vnf(instance_id)

                    # Scaling-in
                    elif scaling_flag < 0:
                        print("scaling in")
                        # If possible to remove VNF instance
                        if num_instances > cfg["instance"]["min_number"]:
                            #sfc_type_index = sfc_info.vnf_instance_ids.index(instance_ids_in_type)
                            index = scaling_target
                            instance_id = instance_ids_in_type[index]
                            instance_ids_in_type.remove(instance_id)
                            sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                            update_sfc(sfc_info)
                            destroy_vnf(instance_id)
                            done = True
                            
                            instance_info_list = []
                            for vnf_array in sfc_info.vnf_instance_ids:
                                instance_info_list.append([])
                                for a_vnf in vnf_array:
                                    instance_info_list[-1].append(ni_mon_api.get_vnf_instance(a_vnf))
                            create_dashboard(instance_info_list,"Auto-scaling-VNF")
                            
        
                    # Maintain
                    else:
                        done = True
                else:
                    pprint("[%s] No scaling because of no target tier!" % (scaler.get_scaling_name()))


            
            # Prepare calculating rewards
            if scaling_flag == 1 and type_name == "firewall":
                print("waiting time for VNF configuration")
                time.sleep(10)
                
            time.sleep(10)
            sfc_info = get_sfc_by_name(scaler.get_sfc_name())
            vnf_info = get_vnf_info(sfc_info)

            service_info = get_service_info(vnf_info, sfc_info, flavors)
            s_prime = state_pre_processor(service_info)

            response_time = 0.0
            for rep in range(0,5):
                response_time = max(response_time, measure_response_time(scaler, "DQN"))
            
            response_time = response_time
            
            print(response_time) ### debug
            if response_time < 0:
                break

            type_instances = get_instances_in_sfc(vnf_info, sfc_info)
            type_status = get_type_status(type_instances, flavors)

            r = reward_calculator(service_info, response_time)
            print("reward")
            print(r) ## for debug

            done_mask = 1.0 if done else 0.0
            transition = (s,a,r,s_prime,done_mask)
            memory.put(transition)

            if memory.size() > required_mem_size:
                train(q, q_target, memory, optimizer, gamma, batch_size)

            if n_epi % print_interval==0 and n_epi != 0:
                print("[%s] Target network updated!" % (scaler.get_scaling_name()))
                q_target.load_state_dict(q.state_dict())

            current_time = dt.datetime.now() #+ dt.timedelta(hours = 24)

            if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
                scaler.set_active_flag(False)

            n_epi = n_epi+1

            if n_epi > 1000:
                scaler.set_active_flag(False)
            if epsilon_value > 0.4 and scaler.has_dataset == False:    
                epsilon_value = epsilon_value - 0.01
            elif scaler.has_dataset == False:
                scaler.set_active_flag(False)
                

            time.sleep(scaler.get_interval())

    # Delete AutoScaler object
    if scaler in scaler_list:
        delete_monitor(scaler)
        scaler_list.remove(scaler)
        pprint("[Expire: %s] DQN Scaling" % (scaler.get_scaling_name()))
    else:
        pprint("[Exit: %s] DQN Scaling" % (scaler.get_scaling_name()))

    q.save_model("./"+scaler.get_scaling_name())



# threshold_scaling(scaler): doing auto-scaling based on threshold
# Input: scaler
# Output: none
def threshold_scaling(scaler):
#    print("27")
    sfc_info = get_sfc_by_name(scaler.get_sfc_name())

    # Target SFC exist
    if sfc_info:
        print("target SFC exists")
        # Initial Processing
        start_time = dt.datetime.now()
        source_client = get_source_client(scaler.get_sfc_name())

        flavors = ni_mon_api.get_vnf_flavors()
        instance_types = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
        #del instance_types[0]

        scaler.set_active_flag(create_monitor(scaler))

        print("before while loop")
        while scaler.get_active_flag():
            response_time = measure_response_time(scaler,"threshold")

            if response_time < 0:
                break

            print("set scaling flag")

            # Set sclaing_flag to show it is out or in or maintain
            if response_time > scaler.get_threshold_out():
                print("[%s] Scaling-out!" % (scaler.get_scaling_name()))
                scaling_flag = 1
            elif response_time < scaler.get_threshold_in():
                print("[%s] Scaling-in!" % (scaler.get_scaling_name()))
                scaling_flag = -1
            else:
                print("[%s] Maintain!" % (scaler.get_scaling_name()))
                scaling_flag = 0

            
            print("action")
            # Scale-in or out
            if scaling_flag != 0:
                sfc_info = get_sfc_by_name(scaler.get_sfc_name())
                print(instance_types)
                vnf_info = get_vnf_info(sfc_info) ########
                type_instances = get_instances_in_sfc(vnf_info, sfc_info) #####

                print("select type")
                # Select Type
                type_status = get_type_status(type_instances, flavors)
                type_index = get_target_type(type_status, source_client.node_id, scaling_flag, True)

                #if tier_index > -1:

                print("if type index")
                if type_index > -1:
                    scaling_target = get_scaling_target(type_status[type_index], source_client.node_id, scaling_flag, True)
                    type_name = instance_types[type_index]
                    instance_ids_in_type = [ instance.id for instance in type_instances[type_index]]
                    num_instances = len(instance_ids_in_type)

                    # Scaling-out
                    if scaling_flag > 0:
                        print("scaling out")
                        # If possible to deploy new VNF instance
                        if num_instances < cfg["instance"]["max_number"]:
                            vnf_spec = get_nfvo_vnf_spec()
                            vnf_spec.vnf_name = sfc_info.sfc_name+ cfg["instance"]["prefix_splitter"] + type_name+ "scaled"
                            vnf_spec.flavor_id = cfg["flavor"][type_name]
                            vnf_spec.image_id = cfg["image"][type_name]
                            vnf_spec.node_name = scaling_target
                            instance_id = deploy_vnf(vnf_spec)

                            # Wait 1 minute until creating VNF instnace
                            limit = 100
                            for i in range(0, limit):
                                time.sleep(2)

                                # Success to create VNF instance
                                if check_active_instance(instance_id):
                                
                                    print("Checked instance actived. Waiting for setup vnf program.(for sfc load balancing)")
                                    time.sleep(150)
                                    
                                    
                                    #sfc_type_index = sfc_info.vnf_instance_ids.index(instance_ids_in_type)
                                    instance_ids_in_type.append(instance_id)
                                    sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                                    update_sfc(sfc_info)
                                    
                                    
                                    instance_info_list = []
                                    for vnf_array in sfc_info.vnf_instance_ids:
                                        instance_info_list.append([])
                                        for a_vnf in vnf_array:
                                            instance_info_list[-1].append(ni_mon_api.get_vnf_instance(a_vnf))
                                    create_dashboard(instance_info_list,"Auto-scaling-VNF")                                    
                                    
                                    break
                                elif i == (limit-1):
                                    destroy_vnf(instance_id)


                    # Scaling-in
                    elif scaling_flag < 0:
                        print("scaling in")
                        # If possible to remove VNF instance
                        if num_instances > cfg["instance"]["min_number"]:

                            #sfc_type_index = sfc_info.vnf_instance_ids.index(instance_ids_in_type)
                            index = scaling_target
                            instance_id = instance_ids_in_type[index]
                            instance_ids_in_type.remove(instance_id)
                            sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                            update_sfc(sfc_info)
                            destroy_vnf(instance_id)
                            
                            instance_info_list = []
                            for vnf_array in sfc_info.vnf_instance_ids:
                                instance_info_list.append([])
                                for a_vnf in vnf_array:
                                    instance_info_list[-1].append(ni_mon_api.get_vnf_instance(a_vnf))
                            create_dashboard(instance_info_list,"Auto-scaling-VNF")                            
                            
                else:
                    #print("else condition in type index")
                    pprint("[%s] No scaling because of no target tier!" % (scaler.get_scaling_name()))
                ####
            command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_http_e2e.sh %s %s %s %s %s" % (
                "ubuntu",
                "dpnm",
                cfg["traffic_controller"]["ip"],
                "141.223.181.190",
                "ubuntu",
                "dpnm",
                150,
                "10.10.20.186"))

            command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"
            print(command)

            # Wait until web server is running
            start_time = dt.datetime.now()
            print(start_time)

            response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
            print(response)
            if response != "":
                f = open("test_e2e.txt", "a+", encoding='utf-8')
                f.write(str(response)+'\n')
                f.close()
                print("write done")

            #print("get current time")
            current_time = dt.datetime.now()

            if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
                scaler.set_active_flag(False)

            #print("sleep")
            time.sleep(scaler.get_interval())


    # Delete AutoScaler object
    if scaler in scaler_list:
        delete_monitor(scaler)
        scaler_list.remove(scaler)
        pprint("[Expire: %s] Threshold Scaling" % (scaler.get_scaling_name()))
    else:
        pprint("[Exit: %s] Thresold Scaling" % (scaler.get_scaling_name()))




def test_measure_response_time():

    try:
        response = "Cannot find test-sfcrs for test"
        deployed_sfcrs = ni_nfvo_sfcr_api.get_sfcrs()

        for sfcr in deployed_sfcrs:
            if sfcr.name.startswith("test-auto-scaling"):
                if get_sfc_by_name(sfcr.name):
                    target_sfcr = sfcr
                    continue

    except:
        return "There is no target sfcr for auto-scaling evaluation"
        
        
    src_ip = (target_sfcr.src_ip_prefix).split('/')[0]
    dst_ip = (target_sfcr.dst_ip_prefix).split('/')[0]
    
  
    cnd_path = os.path.dirname(os.path.realpath(__file__))

    command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_http_e2e.sh %s %s %s %s %s" % (cfg["traffic_controller"]["password"],
                                                                            cfg["traffic_controller"]["username"],
                                                                            cfg["traffic_controller"]["ip"],
                                                                            src_ip,
                                                                            cfg["instance"]["username"],
                                                                            cfg["instance"]["password"],
                                                                            cfg["traffic_controller"]["num_requests"],
                                                                            dst_ip))

    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"


    print(command)

    # Wait until web server is running
    start_time = dt.datetime.now()
    print(start_time)

    while True:
        #print("19-while loop")
        time.sleep(1)
        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        print(response)
        if response != "":
            #print("if")
            pprint("[Test] %s" % (response))
            f = open("test_monitor.txt", "a+", encoding='utf-8')
            f.write(str(response)+'\n')
            f.close()
            print("write done")
            return float(response)
        elif (dt.datetime.now() - start_time).seconds > 60:
            #print("elif")
            return -1




