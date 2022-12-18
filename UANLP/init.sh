#!/bin/sh
pip install wheel
cd /diploma/pymorphy2 && pip install -e .
cd /diploma/pymorphy2-dicts/pymorphy2-dicts-uk && pip install -e .
cd /spark_lp && pip install -e .
useradd spark
usermod -a -G root spark
/opt/bitnami/scripts/spark/run.sh