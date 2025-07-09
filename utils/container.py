import os
from pathlib import Path


def is_docker_running():
    if Path('/.dockerenv').exists():
        return True

    try:
        with open('/proc/1/cgroup', 'r') as f:
            cgroup_content = f.read()
            if 'docker' in cgroup_content or 'containerd' in cgroup_content:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    if os.environ.get('DOCKER_CONTAINER'):
        return True

    return False


def is_apptainer_running():
    singularity_vars = [
        'SINGULARITY_CONTAINER',
        'SINGULARITY_NAME',
        'APPTAINER_CONTAINER',
        'APPTAINER_NAME'
    ]

    for var in singularity_vars:
        if os.environ.get(var):
            return True

    try:
        with open('/proc/1/comm', 'r') as f:
            if 'singularity' in f.read().lower():
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # apptainer_paths = [
    #     '/.singularity.d',
    #     '/singularity',
    #     '/.apptainer.d'
    # ]

    # for path in apptainer_paths:
    #     if Path(path).exists():
    #         return True

    return False


def is_pycharm_running():
    return True if 'PYCHARM_HOSTED' in os.environ else False

def is_vscode_running():
    return True if 'VSCODE_PID' in os.environ else False

def is_jupyter_running():
    return True if 'JUPYTER_SESSION_ID' in os.environ else False

def get_container_info():
    info = {'container_type': None}

    if is_docker_running():
        info['container_type'] = 'docker'
        info['container_details'] = {
            'hostname': os.environ.get('HOSTNAME', 'unknown'),
            'container_id': None
        }
        try:
            with open('/proc/self/cgroup', 'r') as f:
                for line in f:
                    if 'docker' in line:
                        parts = line.strip().split('/')
                        if len(parts) > 2:
                            info['container_details']['container_id'] = parts[-1][:12]
                        break
        except:
            pass
    elif is_apptainer_running():
        info['container_type'] = 'apptainer'
        info['container_details'] = {
            'container_name': os.environ.get('APPTAINER_NAME') or os.environ.get('SINGULARITY_NAME'),
            'container_path': os.environ.get('APPTAINER_CONTAINER') or os.environ.get('SINGULARITY_CONTAINER'),
            'bind_paths': os.environ.get('APPTAINER_BIND') or os.environ.get('SINGULARITY_BIND')
        }
    else:
        info['container_type'] = 'native'
        info['container_details'] = None

    if is_pycharm_running():
        info['ide_type'] = 'pycharm'

    elif is_vscode_running():
        info['ide_type'] = 'vscode'

    elif is_jupyter_running():
        info['ide_type'] = 'jupyter'
    else:
        info['ide_type'] = None

    return info
