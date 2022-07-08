#!/bin/bash

rsync --archive --recursive --verbose -P --times --delete tct@tct-computer.physik.uzh.ch:~/measurements_data/* ~/cernbox/projects/ETL_market_survey/measurements_data
