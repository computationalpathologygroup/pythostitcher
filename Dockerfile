# derive from base docker
FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.8-pt1.10-v1

USER user

ARG CODE_DIR="/home/user/source"

COPY "preprocessing" ${CODE_DIR}
COPY "results" ${CODE_DIR}
COPY "sample_data" ${CODE_DIR}
COPY "src" ${CODE_DIR}
COPY "requirements.txt" "/home/user"

ENV PYTHONPATH "${PYTHONPATH}:/opt/ASAP/bin:${CODE_DIR}/PythoStitcher"

USER root
RUN pip3 install -r /home/user/requirements.txt

USER user

#### configure entrypoint
USER root
#COPY run.sh /root/
ENTRYPOINT ["/bin/bash", "/root/run.sh"]