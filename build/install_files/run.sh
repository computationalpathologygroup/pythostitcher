#!/bin/bash

# Preserve environment variables (slurm and some nvidia variables are set at runtime)
env | grep '^SLURM_\|^NVIDIA_' >> /etc/environment

# Disable python buffer for commands that are executed as user "user"
echo "PYTHONUNBUFFERED=1" >> /etc/environment

# Check if extra arguments were given and execute it as a command.
if [ -z "$1" ]; then
  # Print the command for logging.
  printf "No extra arguments given, running bash\n\n"

  # Run bash
  /bin/bash
else
  # Print the command for logging.
  printf "Executing command: %s\n\n" "$*"

  # Execute the passed command.
  "$@"
fi
