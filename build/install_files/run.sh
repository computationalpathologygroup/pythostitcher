#!/bin/bash

# Preserve environment variables (slurm and some nvidia variables are set at runtime)
env | grep '^SLURM_\|^NVIDIA_' >> /etc/environment

# Disable python buffer for commands that are executed as user "user"
echo "PYTHONUNBUFFERED=1" >> /etc/environment

# Check if extra arguments were given and execute it as a command.
if [ -z "$2" ]; then
  # Print the command for logging.
  printf "No extra arguments given, running jupyter and sshd\n\n"

  # Start the SSH daemon and a Jupyter notebook.
  /usr/sbin/sshd
  sudo --user=user --set-home /bin/bash -c '/usr/local/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='
else
  # Print the command for logging.
  printf "Executing command: %s\n\n" "$*"

  # Execute the passed command.
  sudo --user=user --set-home python3 /home/user/pythostitcher-0.1.1/src/main.py "${@}"
fi
