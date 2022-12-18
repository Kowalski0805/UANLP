FROM bitnami/spark:3.3
USER root
RUN install_packages curl
USER 1001
RUN curl https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.13/3.3.0/spark-sql-kafka-0-10_2.13-3.3.0.jar --output /opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.13-3.3.1.jar