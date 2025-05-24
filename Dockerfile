FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk curl tar bash python3 python3-pip python3-venv wget git \
    && apt-get clean
# Install dependencies

# Install Apache Spark (same version as before)
ENV SPARK_VERSION=3.5.5
RUN wget https://downloads.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz \
    && tar -xvf spark-$SPARK_VERSION-bin-hadoop3.tgz \
    && mv spark-$SPARK_VERSION-bin-hadoop3 /opt/spark \
    && rm spark-$SPARK_VERSION-bin-hadoop3.tgz

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# Set Spark environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

# Create a virtual environment for Python dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies inside virtual environment
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install pyspark findspark numpy pandas

# Ensure Spark scripts are executable
RUN chmod +x $SPARK_HOME/sbin/*.sh && chmod +x $SPARK_HOME/bin/*.sh

# Download required JARs
RUN curl -o /opt/spark/jars/rapids-4-spark_2.12-25.02.1.jar https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/25.02.1/rapids-4-spark_2.12-25.02.1.jar
#curl -o /opt/spark/jars/spark-sql-kafka-0-10_2.12-3.5.5.jar https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.5/spark-sql-kafka-0-10_2.12-3.5.5.jar && \
#    curl -o /opt/spark/jars/spark-token-provider-kafka-0-10_2.12-3.5.5.jar https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.5/spark-token-provider-kafka-0-10_2.12-3.5.5.jar && \
#    curl -o /opt/spark/jars/kafka-clients-3.5.5.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.5/kafka-clients-3.5.5.jar && \
#    curl -o /opt/spark/jars/kafka_2.12-3.5.5.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka_2.12/3.5.5/kafka_2.12-3.5.5.jar && \
#    curl -o /opt/spark/jars/kafka-streams-3.5.5.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-streams/3.5.5/kafka-streams-3.5.5.jar

# Set working directory
WORKDIR /spark_lp

# Set default command
CMD ["/bin/bash"]

#FROM bitnami/spark:3.5.5
#USER root
#RUN install_packages curl
#USER 1001
#RUN curl -o /opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.5.jar https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.5/spark-sql-kafka-0-10_2.12-3.5.5.jar && \
#    curl -o /opt/bitnami/spark/jars/spark-token-provider-kafka-0-10_2.12-3.5.5.jar https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.5/spark-token-provider-kafka-0-10_2.12-3.5.5.jar && \
#    curl -o /opt/bitnami/spark/jars/kafka-clients-3.5.5.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.5/kafka-clients-3.5.5.jar && \
#    curl -o /opt/bitnami/spark/jars/kafka_2.12-3.5.5.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka_2.12/3.5.5/kafka_2.12-3.5.5.jar && \
#    curl -o /opt/bitnami/spark/jars/kafka-streams-3.5.5.jar https://repo1.maven.org/maven2/org/apache/kafka/kafka-streams/3.5.5/kafka-streams-3.5.5.jar
