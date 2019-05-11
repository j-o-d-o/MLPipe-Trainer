#!/bin/bash

##########
## HELP ##
##########
if [[ ( "$1" == "-h" ) || ( "$1" == "--help" ) ]]; then
    echo "Usage: `basename $0` [-h]"
    echo "  Update / Create Conda environment from environment.yml"
    echo
    echo "  -h, --help      Show this help text"
    exit 0
fi

#######################
## PARAMETER PARSING ##
#######################
while :
do
    case "$1" in
        "")
            break
            ;;
        *)
            echo -e "\033[33mWARNING: Argument $1 is unkown\033[0m"
            shift 2 
    esac
done


###########
## START ##
###########
# Create/Update env
if [[ $PATH != $ENV_NAME ]]; then
  # Check if the environment exists
  if [ $? -eq 0 ]; then
    echo ------------  Update Env  ------------
    conda env update -f environment.yml
  else
    # Create the environment and activate
    echo ------------  Create Env  ------------
    conda env create -f environment.yml
  fi
fi

KERAS_BACKEND=tensorflow
