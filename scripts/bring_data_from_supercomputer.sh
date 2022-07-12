#!/bin/bash

rsync --archive --recursive --verbose -P --times sengerm@supercomputer.physik.uzh.ch:~/measurements_data/* ~/cernbox/projects/ETL_market_survey/measurements_data
