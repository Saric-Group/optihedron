#!/bin/bash

#this script requires wget, python 2.7, virtualenv and pip
#OS-X has trouble with the gpu package of lammps, install XCode8 (not 9) and CUDA

#commented out lines allow for the building of a gpu accelerated version. I will clean this process up at some point...

STARTDIR=$(pwd)
cd "$( dirname "${BASH_SOURCE[0]}" )"
WDIR=$(pwd)

while test $# -gt 0; do
        case "$1" in
                -h|--help)
						echo "EMERGENT OPTIMISER INSTALLER"
						echo " "
                        echo "A script for installing the emergent optimiser package. This script will install lammps as a shared local library, and create a python virtual environment with the required pip modules."
                        echo " "
                        echo "You will need bash (with wget), pip, python, python-dev and virtualenv to install the program."
                        echo " "
                        echo "Once installed, use activate.sh to enter the virtual environment, and refer to the help in the python runscript for further direction."
                        echo " "
                        echo "options:"
                        echo "-h, --help                show brief help"
                        exit 0
                        ;;
                *)
                        break
                        ;;
        esac
done



CUDA_HOME=/usr/local/cuda
LAMMPSDIR=./lammps
if [ ! -d $LAMMPSDIR ]; then
	wget -qO- http://lammps.sandia.gov/tars/lammps-16Feb16.tar.gz | tar xvz 
	mv lammps* $LAMMPSDIR
fi
#copy src files to lammps folder
if [ ! -f $LAMMPSDIR/src/liblammps.so ]; then
	cp -rf src/*.h $LAMMPSDIR/src
	cp -rf src/*.cpp $LAMMPSDIR/src
	cd $LAMMPSDIR/src
	make clean-all
	make no-all
	make yes-dipole yes-rigid yes-molecule yes-python yes-opt #yes-gpu
	#cd ${LAMMPSDIR}/lib/gpu && make -f Makefile.mpi CUDA_LIB="-L${CUDA_HOME}/lib"
	make -j4 mpi #LMP_INC="-DLAMMPS_PNG -DLAMMPS_JPEG -DLAMMPS_FFMPEG -DLAMMPS_EXCEPTIONS" JPG_LIB="-lpng -ljpeg" gpu_SYSPATH="-L${CUDA_HOME}/lib"
	make -j4 mpi mode=shlib #LMP_INC="-DLAMMPS_PNG -DLAMMPS_JPEG -DLAMMPS_FFMPEG -DLAMMPS_EXCEPTIONS" JPG_LIB="-lpng -ljpeg" gpu_SYSPATH="-L${CUDA_HOME}/lib"
	cd "$WDIR"
fi
if [ ! -d venv ]; then
	virtualenv -p `which python2.7` venv
fi
source activate.sh
if [ "`which python2.7`" != "${WDIR}/venv/bin/python2.7" ]; then
echo "ERROR: Virtualenv is not there or couldn't activate. Aborting so LAMMPS doesn't install on system python. Re-run this script to finish installation after sorting out whatever is wrong with virtualenv."
exit
fi
source pipinstall.sh

cd $LAMMPSDIR/python
python install.py

cd "${STARTDIR}"
echo "done!"
