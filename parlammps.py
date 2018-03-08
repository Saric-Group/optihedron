from multiprocessing import Pool, TimeoutError, Process
import time
import os
import subprocess
import plammps
import sys

from lammps import lammps
import signal
import traceback

from tools import templatetools as tt

n = 1
nWorkers = 8



class Signal(Exception):
    """
    This exception is raise by the signal handler.
    """
    pass


class Timeout(Exception):
    """
    This exception is raised when the command exceeds the defined timeout
    duration and the command is killed.
    """
    def __init__(self, cmd, timeout):
        self.cmd = cmd
        self.timeout = timeout

    def __str__(self):
        return "Command '%s' timed out after %d second(s)." % \
               (self.cmd, self.timeout)


class Retcode(Exception):
    """
    This exception is raise when a command exits with a non-zero exit status.
    """
    def __init__(self, cmd, retcode, output=None):
        self.cmd = cmd
        self.retcode = retcode
        self.output = output

    def __str__(self):
        return "Command '%s' returned non-zero exit status %d" % \
               (self.cmd, self.retcode)


def alarm_handler(signum, frame):
    raise Signal


def execute(cmd, timeout=None):
    """
    Execute a command in the default shell. If a timeout is defined the command
    will be killed if the timeout is exceeded and an exception will be raised.
    Inputs:
        cmd     (str): Command to execute
        timeout (int): Command timeout in seconds
    Outputs:
        output (str): STDOUT/STDERR
    """
    # Define the timeout signal
    if timeout:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)

    try:
        # Execute the command and wait for the subprocess to terminate
        # STDERR is redirected to STDOUT
        phandle = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        
        # Read the stout/stderr line by line until subprocess is done
        retcode = None
        while (retcode == None):
            for stdout_line in iter(phandle.stdout.readline, b''):
                yield stdout_line
            retcode = phandle.poll()
        
        # Read the stdout/sterr buffers and retcode
        #output, error = phandle.communicate()
    except Signal:
        # Kill the running process
        phandle.kill()
        raise Timeout(cmd=cmd, timeout=timeout)
    except:
        raise
    else:
        # Possible race condition where alarm isn't disabled in time
        signal.alarm(0)

    # Raise an exception if the command exited with non-zero exit status
    # if retcode:
    #     print("MPI exited with code 1, did you forget to finalize?")
    #     raise Retcode(cmd, retcode, output=output)

    #return output, error
    #yield retcode

def runSim(script,np,timeout,silent=True):
    try:
        for stdout_line in execute(['mpirun','-np',str(np),'./venv/bin/python','./plammps.py','-s',str(script)],timeout):
            if (silent):
                pass
            else:
                print stdout_line,
        return True
    except TimeoutError:
        print('Process timed out')
    except Exception as e:
        print(e)
        traceback.print_exc()
    return False

def createPbs(script,wd,np,name,rundir,mpirun):
    try:
        pbs = os.path.join(wd,"qsubtask.pbs")
        with open(pbs) as templateFile:
            content = templateFile.read()
            if os.path.isfile("modules"): 
                with open("modules") as moduleList:
                    modLine = ""
                    modules=moduleList.read().splitlines()
                    if len(modules) > 0:
                        modLine+="module load "
                    for m in modules:
                        modLine += m + ' '
                    content = content.replace('_MODULES_',modLine)
            else:
                content = content.replace('_MODULES_','')
            content = content.replace('_DIR_',str(wd))
            content = content.replace('_PRE_','mpirun -np _CORES_') if mpirun else content.replace('_PRE_ ','')
            content = content.replace('_CORES_',str(np))
            content = content.replace('_SCRIPT_',str(script))
            pbsPath = os.path.join(rundir,name+".pbs")
            with open(pbsPath, "w") as text_file:
                text_file.write(content)
            return pbsPath
    except Exception as e:
        print(e)
        traceback.print_exc()

def runSimSerial(script):
    try:
        plammps.startScript(script)
        return True
    except:
        print('LAMMPS process crashed')
    return False

def runSims(scripts,np,timeout):
    processes = []
    for s in scripts:
        print(s)
        p = Process(target=runSim, args=(s,np,timeout))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
