#!/usr/bin/env python3
import subprocess
import sys
import os

tasks = [
    {
        "name": "Tuesday",
        "cmd": [
            ".venv/bin/python3", "scripts/sparsify_pcap_generalized.py",
            "--spec", "specs/feature_spec_v5_port_agnostic.json",
            "--input", "datasets/CICIDS2017/tuesday/Tuesday-WorkingHours.pcap",
            "--train-output", "datasets/CICIDS2017/tuesday/v5TuesdayTrain.txt",
            "--test-output", "datasets/CICIDS2017/tuesday/v5TuesdayTest.txt",
            "--labels-csv", "/home/bizio/Repos/HashingDeepLearning/dataset/CICIDS2017_improved/tuesday.csv",
            "--label-map", "configs/label_maps/label_map_cicids2017_multiclass.json",
            "--flow-manifest", "audit_reports/flow_manifest_tuesday.json"
        ],
        "log": "logs/extract_tuesday.log"
    },
    {
        "name": "Wednesday",
        "cmd": [
            ".venv/bin/python3", "scripts/sparsify_pcap_generalized.py",
            "--spec", "specs/feature_spec_v5_port_agnostic.json",
            "--input", "datasets/CICIDS2017/wednesday/Wednesday-workingHours.pcap",
            "--train-output", "datasets/CICIDS2017/wednesday/v5WednesdayTrain.txt",
            "--test-output", "datasets/CICIDS2017/wednesday/v5WednesdayTest.txt",
            "--labels-csv", "/home/bizio/Repos/HashingDeepLearning/dataset/CICIDS2017_improved/wednesday.csv",
            "--label-map", "configs/label_maps/label_map_cicids2017_multiclass.json",
            "--flow-manifest", "audit_reports/flow_manifest_wednesday.json"
        ],
        "log": "logs/extract_wednesday.log"
    },
    {
        "name": "Thursday",
        "cmd": [
            ".venv/bin/python3", "scripts/sparsify_pcap_generalized.py",
            "--spec", "specs/feature_spec_v5_port_agnostic.json",
            "--input", "datasets/CICIDS2017/thursday/Thursday-WorkingHours.pcap",
            "--train-output", "datasets/CICIDS2017/thursday/v5ThursdayTrain.txt",
            "--test-output", "datasets/CICIDS2017/thursday/v5ThursdayTest.txt",
            "--labels-csv", "/home/bizio/Repos/HashingDeepLearning/dataset/CICIDS2017_improved/thursday.csv",
            "--label-map", "configs/label_maps/label_map_cicids2017_multiclass.json",
            "--flow-manifest", "audit_reports/flow_manifest_thursday.json"
        ],
        "log": "logs/extract_thursday.log"
    },
    {
        "name": "Friday",
        "cmd": [
            ".venv/bin/python3", "scripts/sparsify_pcap_generalized.py",
            "--spec", "specs/feature_spec_v5_port_agnostic.json",
            "--input", "datasets/CICIDS2017/friday/Friday-WorkingHours.pcap",
            "--train-output", "datasets/CICIDS2017/friday/friday_v5_flow_train.txt",
            "--test-output", "datasets/CICIDS2017/friday/friday_v5_flow_test.txt",
            "--labels-csv", "/home/bizio/Repos/HashingDeepLearning/dataset/CICIDS2017_improved/friday.csv",
            "--label-map", "configs/label_maps/label_map_cicids2017_multiclass.json",
            "--flow-manifest", "audit_reports/flow_manifest_friday.json"
        ],
        "log": "logs/extract_friday.log"
    }
]

os.makedirs("logs", exist_ok=True)

for t in tasks:
    log_f = open(t["log"], "w")
    proc = subprocess.Popen(
        t["cmd"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # setsid: completely detached from terminal session
        close_fds=True
    )
    print(f"Launched {t['name']} extraction with PID: {proc.pid} (logging to {t['log']})")

