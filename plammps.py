from lammps import lammps
import argparse

import os.path

def runLammps(script):
    lmp = lammps()
    try:
        lmp.file(script)
    except:
        print("error")
    lmp.close()
    return

def startScript(script):
    print(script)
    if(os.path.isfile(script)):
        runLammps(script)
        return True
    print("file does not exist")
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s','--script',
                    help='address of lammps script to open', required=True)
    args = parser.parse_args()
    SCRIPT = args.script
    startScript(SCRIPT)