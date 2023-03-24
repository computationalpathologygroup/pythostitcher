This folder provides the build information to build your own PythoStitcher container locally. The install_files folder contains two components:
1. requirements.txt - this is a precompiled list with dependencies which are tested to be compatible
2. run.sh - this is the bash file to execute when you start your Docker

The provided run.sh file will give you two options for running the PythoStitcher Docker:
1. Automatically. This mode engages when you provide the docker with the PythoStitcher input arguments mentioned on the Readme of the repository. It will then just run the packaged PythoStitcher code directly and is the recommended way of using PythoStitcher.
2. Interactive. If you run the PythoStitcher Docker without any input arguments, it will launch an interactive instance which you can couple to your IDE of choice. This will allow you to run a local, modified version of PythoStitcher according to your needs. This is the advanced way of using the container and only recommended if you need to make any alterations to the code.