---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Plankton ML - workflow diagrams
---

# Workflow Diagrams

Views of the flow of data from the imaging instrument to cloud-accessible storage

### As is

Data saved during a session with the microscope is downloaded onto a USB key, then uploaded from a researcher's laptop into a shared storage area on a site-specific SAN.

Later, a data scientist logs into a virtual machine in the on-premise "private cloud" and runs more than one script to read the data, process it for analysis, and then upload to s3 storage hosted at JASMIN. Authorisation in this chain requires personal credentials.

<object data="as_is/instrument_to_store.svg" type="image/svg+xml">
</object>

There are file naming conventions including metadata which doesn't follow the same path as the data, and there are spatio-temporal properties of the samples which could be recorded.

### Could be 

PC that drives the instrument is connected to the storage network, but not the internet (for security standards compliance reasons). What are the current precedents for either directly saving output to shared storage, or a watcher process that either pulls or pushes data from a lab PC to networked storage?

Automated workflow (could be Apache Airflow or Beam based - FDRI project is trialling components) which watches for new source data, distributes the preprocessing with Dask or Spark if necessary, and publishes analysis-ready data _and metadata_ to cloud storage, continuously.

<object data="could_be/instrument_to_store.svg" type="image/svg+xml">
</object>


