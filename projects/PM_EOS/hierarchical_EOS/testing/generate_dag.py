import os
import sys
import numpy as np

# set up list for strings to be written to file and adding the first check submit file
# This job is simply so that we can ensure the first job succeeds
dag_lines = [f"JOB first /home/ethan.payne/projects/calibration_marginalization/event_set_submission/first_check_submit.sub"]

launch_string = 'PARENT first CHILD '


for id in range(int(1e3)):
            
    dag_lines += ['\n',
        f"JOB {id} /home/ethan.payne/projects/GWPhotonCounting/projects/PM_EOS/hierarchical_EOS/testing/injection_submit.sub",
        f"VARS {id} "\
        f"id=\"{id}\" "]

    launch_string += f"{id} "

dag_lines.append(launch_string)

# Write out to file
with open(f'injection_submission.dag', 'w+') as filehandle:
    for listitem in dag_lines:
        filehandle.write('%s\n' % listitem)
