#!/bin/bash

##################################################################
# START_IP_HEADER                                                #
#                                                                #
# Written by Francois Fleuret                                    #
# Contact <francois.fleuret@unige.ch> for comments & bug reports #
#                                                                #
# END_IP_HEADER                                                  #
##################################################################

# set -e
# set -o pipefail

#prefix="--nb_train_samples=1000 --nb_test_samples=100 --batch_size=25 --nb_epochs=2 --max_percents_of_test_in_train=-1 --model=17K"
prefix="--nb_epochs=25"

for task in byheart learnop guessop twotargets addition picoclvr maze snake stack expr rpl
do
    [[ ! -d results_${task} ]] && ./src/main.py ${prefix} --task=${task}
done

