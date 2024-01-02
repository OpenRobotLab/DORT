import time
import re
import os.path as osp
import numpy as np
import subprocess
import shutil
import os
#from tensorflow.python.client import device_lib
from datetime import datetime
def preprocess_log_script(log_script):
    filter = re.findall("configs\/.+\/(.+)", log_script)
    if len(filter) > 0:
        log_script = filter[0]
        log_script = log_script.replace("--ceph", "")
        log_script = log_script.replace("--cfg-options", "")
        log_script = log_script.replace("--", "-")
    else:
        log_script = log_script.replace("bash_._train.sh_configs_","").replace("--config_configs_", "")
        log_script = log_script.replace("tools_dist_train.sh_configs_", "")
        log_script = log_script.replace("tools-", "")
        log_script = log_script.replace(".-", "")
        log_script = log_script.replace(".py", "")
        log_script = log_script.replace("--ceph-", "")
        log_script = log_script.replace(".txt", "")
        log_script = log_script.replace("--", "-")

    return log_script


def check_gpus():
    '''
    GPU available check
    reference : http://feisky.xyz/machine-learning/tensorflow/gpu_list.html
    '''
# =============================================================================
#     all_gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
# =============================================================================
    first_gpus = os.popen('nvidia-smi --query-gpu=index --format=csv,noheader').readlines()[0].strip()
    if not first_gpus=='0':
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True

def parse(line,qargs):
    '''
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=[]):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    results = [parse(line,qargs) for line in results]
    for idx, result in enumerate(results):
        result['memory.used'] = int(result['memory.total']) - int(result['memory.free'])
        results[idx] = result
    return results

def by_power(d):
    '''
    helper function fo sorting gpus by power
    '''
    power_infos=(d['power.draw'],d['power.limit'])
    if any(v==1 for v in power_infos):
        print('Power management unable for GPU {}'.format(d['index']))
        return 1
    return float(d['power.draw'])/d['power.limit']

class GPUManager():
    '''
    qargs:
        query arguments
    A manager which can list all available GPU devices
    and sort them and choice the most free one.Unspecified
    ones pref.
    GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
    最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
    优先选择未指定的GPU。
    '''
    def __init__(self,qargs=[]):
        '''
        '''
        self.qargs=qargs
        self.gpus=query_gpu(qargs)
        for gpu in self.gpus:
            gpu['specified']=False
        self.gpu_num=len(self.gpus)

    def _sort_by_memory(self,gpus,by_size=False):
        if by_size:
            print('Sorted by free memory size')
            return sorted(gpus,key=lambda d:d['memory.free'],reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(gpus,key=lambda d:float(d['memory.free'])/ d['memory.total'],reverse=True)

    def _sort_by_power(self,gpus):
        return sorted(gpus,key=by_power)

    def _sort_by_custom(self,gpus,key,reverse=False,qargs=[]):
        if isinstance(key,str) and (key in qargs):
            return sorted(gpus,key=lambda d:d[key],reverse=reverse)
        if isinstance(key,type(lambda a:a)):
            return sorted(gpus,key=key,reverse=reverse)
        raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

    def auto_choice(self,mode=0):
        '''
        mode:
            0:(default)sorted by free memory size
        return:
            a TF device object
        Auto choice the freest GPU device,not specified
        ones
        自动选择最空闲GPU
        '''
        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

        if mode==0:
            print('Choosing the GPU device has largest free memory...')
            chosen_gpu=self._sort_by_memory(unspecified_gpus,True)[0]
        elif mode==1:
            print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
        elif mode==2:
            print('Choosing the GPU device by power...')
            chosen_gpu=self._sort_by_power(unspecified_gpus)[0]
        else:
            print('Given an unaviliable mode,will be chosen by memory')
            chosen_gpu=self._sort_by_memory(unspecified_gpus)[0]
        chosen_gpu['specified']=True
        index=chosen_gpu['index']
        print('Using GPU {i}:\n{info}'.format(i=index,info='\n'.join([str(k)+':'+str(v) for k,v in chosen_gpu.items()])))
        return tf.device('/gpu:{}'.format(index))

    def choose_no_task_gpu(self):
        for old_infos,new_infos in zip(self.gpus,query_gpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus=[gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

        gpu_av = []
        gpu_mem = []

        for i in range(len(unspecified_gpus)):
            if unspecified_gpus[i]['memory.used'] < 10000:
                gpu_av.append(i)
                gpu_mem.append(unspecified_gpus[i]['memory.used'])
        gpu_av = [x for _, x in sorted(zip(gpu_mem, gpu_av), reverse=False)]
        return gpu_av



def check_available(gpu_av, max_gpu_av):
    if max_gpu_av == "":
        return gpu_av
    else:
        gpu_av = [i for i in gpu_av if i in max_gpu_av]
        return gpu_av

def split_script(mission):
    scripts = mission.split("#")
    if len(scripts) > 1:
        script, gpu_need = scripts
        gpu_need = "#" + gpu_need
    else:
        script = scripts[0]
        gpu_need = ""

    scripts = script.split("--cfg-options")
    if len(scripts) > 1:
        script, cfg_options = scripts
        cfg_options = "--cfg-options" + cfg_options
    else:
        script = scripts[0]
        cfg_options = ""
    return script, cfg_options, gpu_need

def repeat_mission(mission_queue, repeat_time):
    new_mission_queue = []
    for i in range(int(repeat_time)):
        for mission in mission_queue:
            script, cfg_options, gpu_need = split_script(mission)

            script = script + "--seed {} ".format(i+1)
            script = script + cfg_options + gpu_need
            new_mission_queue.append(script)
    return new_mission_queue

def add_extra_arg(mission_queue, extra_arg):
    new_mission_queue = []
    for mission in mission_queue:
        script, cfg_options, gpu_need = split_script(mission)
        script = script + " " + extra_arg
        new_mission_queue.append(script + cfg_options+ gpu_need)

    return new_mission_queue


def find_gpu(script):
    gpu_needs_wrapper =  re.findall("# ([0-9]*)", script)
    if len(gpu_needs_wrapper) > 0:
        gpu_needs = int(gpu_needs_wrapper[0])
        script = script.replace(f"# {gpu_needs}", "")
        print(f"processed script: {script}")
        return gpu_needs, script
    gpu_needs = re.findall("--num-gpus ([0-9])", script)
    if len(gpu_needs) == 0:
        return 1, script
    elif len(gpu_needs) == 1:
        return int(gpu_needs[0]), script
    else:
        print("strange gpu num", gpu_needs)
        return 1, script


def run_in_local(mission_queue, sup_dir, options):
    gm=GPUManager()
    for mission in mission_queue:
        print(mission)
    max_gpu_av = options.available_gpu
    if max_gpu_av == "":
        print("All the gpu can be used")
    else:
        max_gpu_av = max_gpu_av.split(",")
        max_gpu_av = [int(i) for i in max_gpu_av]
        print(f"Only {max_gpu_av} can be used.")
    localtime = time.asctime(time.localtime(time.time()))

    print("running script:",  mission_queue)
    basename = os.path.basename(os.getcwd())
    home_dir = os.path.expanduser('~')
    running_cmd = []
    p = []
    while len(mission_queue) > 0:
        gpu_av = gm.choose_no_task_gpu()
        gpu_av = check_available(gpu_av, max_gpu_av)

        script = mission_queue.pop(0)

        gpu_needs, script = find_gpu(script)
        if options.ceph is True:
            script, cfg_options, gpu_need = split_script(script)

            script = script + "--ceph "
            script = script + cfg_options


        while len(gpu_av) < gpu_needs:
            gpu_av = gm.choose_no_task_gpu()
            gpu_av = check_available(gpu_av, max_gpu_av)

            print("Keep looking @ %s"%(localtime))
            print("Remaining mission: ", mission_queue)

            time.sleep(300)

        print("begin running")
        gpu_index = gpu_av[:gpu_needs]

        #gpu_index = np.random.choice(gpu_av, gpu_needs, replace=False)
        gpu_index = [str(i) for i in gpu_index]

        localtime = time.asctime(time.localtime(time.time()))

        if not options.tacc:
            log_script_dir = osp.join(home_dir, "exp_logs")
        else:
            log_script_dir = osp.join(home_dir, "USERDIR", "exp_logs")
        log_script_dir = osp.join(log_script_dir, basename, "logs")

        os.makedirs(log_script_dir, exist_ok=True)

        log_script = script.replace("python ", "").replace(" ", "-")
        log_script = preprocess_log_script(log_script)
        if log_script[0] == ".":
            log_script = log_script[1:]

        now = datetime.now()
        time_string = now.strftime("%m%d-%H%M")
        day_string = now.strftime("%m%d")

        log_script = log_script + time_string
        log_script = os.path.join(log_script_dir, day_string) + "/" + log_script

        os.makedirs(os.path.dirname(log_script), exist_ok=True)

        if options.tacc is True:
            if "--opts" in script:
                script, opts = script.split("--opts ")
            else:
                opts = ""
            script = script + f"--work-dir /mnt/home/qlianab/USERDIR/exp_logs/{basename}/ --tacc"
            if len(opts) > 0:
                script = script + " --ops " + opts
            script = script + " >> " + log_script + " 2>&1"
        else:
            script = script + " >> " + log_script + " 2>&1"
        gpu_index = ",".join(gpu_index)
        cmd_ = "CUDA_VISIBLE_DEVICES=" +gpu_index + " " + script

        p.append(subprocess.Popen(cmd_, shell=True))

        print('mission: %s\n run on gpu: %s\nStarted: %s\n'%(cmd_, gpu_index, localtime))
        time.sleep(300)



        new_p = [] # save the process that is still training
        print("Current running task")
        for i in range(len(p)):
            if p[i].poll() != None:
                log_file = p[i].args.split(">>")[1]
                log_file = log_file.replace("2>&1", "")
                log_file = log_file.strip()
                if len(log_file) == 0:
                    continue
                new_log_file = log_file.replace("logs/", "logs/finished")

                new_log_dir = os.path.dirname(new_log_file)
                os.makedirs(new_log_dir, exist_ok=True)
                try:
                    shutil.copyfile(osp.join(sup_dir, log_file.replace('"', '')), osp.join(sup_dir, new_log_file))
                except:
                    pass
            else:
                print(f"Command: {p[i].args}, pid: {p[i].pid}")

                new_p.append(p[i])
        p = new_p
    for i in range(len(p)):
        print("Still running: ")
        print(f"Command: {p[i].args}, pid: {p[i].pid}")
        p[i].wait()

    print("Mission Complete! Checking GPU process over")

def preprocess_slurm_queue(misson_queue, options):
    new_mission_queue = []
    partion = options.slurm
    if partion is None or partion is False:
        partion = "robot"
    for mission in misson_queue:
        # name = re.findall(re.findall("configs.+py ", b))
        name =re.findall("configs.+py ", mission)
        if len(name) > 0:
            name = osp.basename(name[0])
        else:
            name = "mmdet3d"
        mission = mission.replace("./tools/dist_train.sh",\
                 "./tools/custom_slurm_train.sh {} {}".format(partion, name))
        mission  = mission.split("#")[0]
        new_mission_queue.append(mission)

    return new_mission_queue


def run_in_slurm(mission_queue, sup_dir, options):
    # slurm_bash = options.slurm
    basename = os.path.basename(os.getcwd())
    home_dir = os.path.expanduser('~')
    mission_queue = preprocess_slurm_queue(mission_queue, options)

    for mission in mission_queue:
        print(mission)
    for idx, mission in enumerate(mission_queue):
        # print("current mission: {} {}".format())
        mission = mission.split("#")[0]
        if options.ceph is True:
            script, cfg_options, gpu_need = split_script(mission)

            script = script + "--ceph "
            script = script + cfg_options
        if not options.tacc:
            log_script_dir = osp.join(home_dir, "exp_logs")
        else:
            log_script_dir = osp.join(home_dir, "USERDIR", "exp_logs")
        log_script_dir = osp.join(log_script_dir, basename, "logs")
        os.makedirs(log_script_dir, exist_ok=True)
        log_script = script.replace("python ", "").replace(" ", "-")
        log_script = preprocess_log_script(log_script)
        if log_script[0] == ".":
            log_script = log_script[1:]
        now = datetime.now()
        time_string = now.strftime("%m%d-%H%M")
        day_string = now.strftime("%m%d")

        log_script = log_script + time_string
        log_script = os.path.join(log_script_dir, day_string) + "/" + log_script
        os.makedirs(os.path.dirname(log_script), exist_ok=True)

        script = "nohup " + script + " >> " + log_script + " 2>&1 &"
        print("running {} {}".format(idx, script))
        os.system(script)
        time.sleep(5)
