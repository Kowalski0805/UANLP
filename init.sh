#!/bin/sh
#pip install wheel
##cd /diploma/pymorphy2 && pip install -e .
##cd /diploma/pymorphy2-dicts/pymorphy2-dicts-uk && pip install -e .
#cd /spark_lp && pip install -e .
#useradd spark
#usermod -a -G root spark
#/opt/bitnami/scripts/spark/run.sh

# Ensure CUDA environment is set
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Check if GPU is accessible inside the container
nvidia-smi || echo "Warning: No NVIDIA GPU detected!"
# Activate Python virtual environment
export PATH="/opt/venv/bin:$PATH"

# Install Python dependencies inside virtual environment
/opt/venv/bin/pip install -e /spark_lp
/opt/venv/bin/pip install \
                      --extra-index-url=https://pypi.nvidia.com \
                      "cudf-cu12==25.2.*"

# Create Spark user if not exists
#if ! id "spark" &>/dev/null; then
    useradd -m spark
    usermod -a -G root spark
#fi

# 🔥 Set Spark GPU configs
echo "spark.worker.resource.gpu.amount=1" >> /opt/spark/conf/spark-defaults.conf
echo "spark.task.resource.gpu.amount=1" >> /opt/spark/conf/spark-defaults.conf
echo "spark.executor.resource.gpu.amount=1" >> /opt/spark/conf/spark-defaults.conf
echo "spark.worker.resource.gpu.discoveryScript=/spark_lp/getGpusResources.sh" >> /opt/spark/conf/spark-defaults.conf


# Define Spark Home
export SPARK_HOME="/opt/spark"
export PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

# Start Spark based on mode
if [ "$SPARK_MODE" = "master" ]; then
    echo "Starting Spark Master..."
    $SPARK_HOME/sbin/start-master.sh
    tail -f /opt/spark/logs/spark--org.apache.spark.deploy.master.Master-*.out
elif [ "$SPARK_MODE" = "worker" ]; then
    echo "Starting Spark Worker..."
    $SPARK_HOME/sbin/start-worker.sh spark://spark:7077
    tail -f /opt/spark/logs/spark--org.apache.spark.deploy.worker.Worker-*.out
else
    echo "Error: SPARK_MODE must be 'master' or 'worker'."
    exit 1
fi
